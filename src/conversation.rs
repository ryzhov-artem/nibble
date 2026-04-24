use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub max_history_tokens: usize,
    pub system_prompt: String,
}

impl Conversation {
    pub fn new(system_prompt: String, max_history_tokens: usize) -> Self {
        let system_msg = Message {
            role: MessageRole::System,
            content: system_prompt.clone(),
        };
        Self {
            messages: vec![system_msg],
            max_history_tokens,
            system_prompt,
        }
    }

    pub fn add_user_message(&mut self, content: String) {
        self.messages.push(Message { role: MessageRole::User, content });
    }

    pub fn add_assistant_message(&mut self, content: String) {
        self.messages.push(Message { role: MessageRole::Assistant, content });
    }

    pub fn format_prompt(&self, _tokenizer: &Tokenizer) -> candle::Result<String> {
        let mut prompt = String::new();
        for msg in &self.messages {
            match msg.role {
                MessageRole::System => {
                    prompt.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content));
                }
                MessageRole::User => {
                    prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content));
                }
                MessageRole::Assistant => {
                    prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content));
                }
            }
        }
        prompt.push_str("<|assistant|>\n");
        Ok(prompt)
    }

    pub fn apply_sliding_window(&mut self, tokenizer: &Tokenizer) -> candle::Result<()> {
        let full_prompt = self.format_prompt(tokenizer)?;
        let tokens = tokenizer
            .encode(full_prompt, false)
            .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
            .get_ids()
            .len();

        if tokens <= self.max_history_tokens {
            return Ok(());
        }

        let mut kept_messages = vec![self.messages[0].clone()];
        let mut current_tokens = tokenizer
            .encode(self.system_prompt.clone(), false)
            .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
            .get_ids()
            .len();

        for msg in self.messages.iter().skip(1).rev() {
            let msg_tokens = tokenizer
                .encode(msg.content.clone(), false)
                .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
                .get_ids()
                .len();

            if current_tokens + msg_tokens > self.max_history_tokens {
                break;
            }
            kept_messages.insert(1, msg.clone());
            current_tokens += msg_tokens;
        }

        let removed = self.messages.len() - kept_messages.len();
        if removed > 0 {
            println!("\nContext trimmed: kept {} messages, removed {} old", kept_messages.len(), removed);
        }
        self.messages = kept_messages;
        Ok(())
    }
}