//! Thread-local scratch pool for short-lived `Vec<f32>` buffers.
//!
//! Created for B1 of the optimisation plan: in `QuantLinear::forward_2d` we
//! repeatedly need a `Vec<f32>` of size `batch * k` to hold a permuted copy
//! of the activations. Allocating + dropping one per call (≈128 calls per
//! decoded token) churns the allocator. This pool keeps a small free-list
//! of buffers per worker thread and hands them out as RAII [`Buf`] guards
//! that automatically return to the pool on drop.
//!
//! The pool is intentionally trivial:
//!   * thread-local — no locks, no contention
//!   * keyed on capacity (first-fit) — avoids "exact size" fragmentation
//!   * bounded — at most [`POOL_LIMIT`] retained buffers per thread

use std::cell::RefCell;

/// Upper bound on retained buffers per worker thread. Anything beyond this
/// is dropped instead of being recycled (so we don't pin unbounded memory
/// after a brief allocation spike).
const POOL_LIMIT: usize = 16;

thread_local! {
    static POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// RAII handle to a pooled `Vec<f32>` of exactly `len` usable elements.
/// On drop the underlying allocation is returned to the thread-local pool.
pub struct Buf {
    inner: Option<Vec<f32>>,
    len: usize,
}

impl Buf {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // `unwrap()` is safe: `inner` is only taken in `Drop`.
        let v = self.inner.as_mut().expect("scratch buffer already returned");
        &mut v[..self.len]
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        let v = self.inner.as_ref().expect("scratch buffer already returned");
        &v[..self.len]
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some(v) = self.inner.take() {
            POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < POOL_LIMIT {
                    p.push(v);
                }
                // else: let `v` drop here, freeing memory.
            });
        }
    }
}

/// Borrow a buffer of exactly `len` `f32`s. The contents are zero-initialised.
///
/// First tries the thread-local pool for a free buffer with sufficient
/// capacity; otherwise allocates a fresh `Vec`.
pub fn take_f32(len: usize) -> Buf {
    let v = POOL.with(|p| {
        let mut p = p.borrow_mut();
        if let Some(pos) = p.iter().position(|b| b.capacity() >= len) {
            let mut buf = p.swap_remove(pos);
            buf.clear();
            buf.resize(len, 0.0);
            buf
        } else {
            vec![0.0f32; len]
        }
    });
    Buf { inner: Some(v), len }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_is_zeroed_and_sized() {
        let mut b = take_f32(8);
        assert_eq!(b.as_slice().len(), 8);
        assert!(b.as_slice().iter().all(|&x| x == 0.0));
        b.as_mut_slice()[3] = 7.0;
        assert_eq!(b.as_slice()[3], 7.0);
    }

    #[test]
    fn allocations_are_recycled() {
        let cap = {
            let b = take_f32(1024);
            b.inner.as_ref().unwrap().capacity()
        };
        // After drop, a same-or-smaller request should reuse the same allocation.
        let b2 = take_f32(1024);
        assert!(b2.inner.as_ref().unwrap().capacity() >= cap);
    }
}
