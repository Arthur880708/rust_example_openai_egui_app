use eframe::egui;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use dotenvy::dotenv;
use std::env;
use tokio::runtime::Runtime;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: MessageContent,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

async fn analyze_data_with_gpt4(
    api_key: &str,
    instructions: &str,
    data_json: &str,
) -> Result<String, reqwest::Error> {
    let client = Client::new();

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: instructions.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: data_json.to_string(),
        },
    ];

    let request_body = OpenAIRequest {
        model: "gpt-4".to_string(),
        messages,
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await?;

    let response_body: OpenAIResponse = response.json().await?;
    let advice = &response_body.choices[0].message.content;

    Ok(advice.clone())
}

struct MyApp {
    response: Arc<Mutex<String>>,
    my_message: String,
    api_key: String,
    data_json: String,
    rt: Runtime,
}

impl Default for MyApp {
    fn default() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("API key not found in .env file");

        let my_message = fs::read_to_string("instruction.md")
            .expect("Unable to read instruction.md file");

        Self {
            api_key,
            my_message,
            data_json: String::new(),
            response: Arc::new(Mutex::new(String::new())),
            rt: Runtime::new().unwrap(),  // 비동기 런타임 생성
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 폰트 로드
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "my_font".to_owned(),
            egui::FontData::from_static(include_bytes!("NanumGothic-Regular.ttf")),
        );

        fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "my_font".to_owned());
        fonts.families.entry(egui::FontFamily::Monospace).or_default().push("my_font".to_owned());

        ctx.set_fonts(fonts);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("OpenAI 카 딜러");

            ui.label("질문 :");
            ui.text_edit_multiline(&mut self.data_json);

            if ui.button("질문 보내기").clicked() {
                let my_message = self.my_message.clone();
                let data_json = self.data_json.clone();
                let api_key = self.api_key.clone();
                let response_clone = Arc::clone(&self.response);
                
                self.rt.spawn(async move {
                    match analyze_data_with_gpt4(&api_key, &my_message, &data_json).await {
                        Ok(advice) => {
                            let mut res = response_clone.lock().await;
                            *res = advice;
                        }
                        Err(e) => {
                            let mut res = response_clone.lock().await;
                            *res = format!("Error: {}", e);
                        }
                    }
                });
            }

            ui.label("딜러의 대답 :");
            
            let response = self.response.clone();
            let response_text = self.rt.block_on(async {
                let res = response.lock().await;
                res.clone()
            });

            ui.add(
                egui::TextEdit::multiline(&mut response_text.as_str())
                    .font(egui::TextStyle::Monospace)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(20),
            );

            if ui.button("종료").clicked() {
                _frame.close();
            }
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "OpenAI 카 딜러(Ver.1.0)",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    );
    Ok(())
}



// Unit Test
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use httpmock::MockServer;
    use httpmock::Method::POST;

    #[tokio::test]
    async fn test_analyze_data_with_gpt4() {
        let server = MockServer::start();
        
        let mock_response_body = r#"
        {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response from GPT-4."
                    }
                }
            ]
        }"#;

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .header("Authorization", "Bearer mock_api_key");
            then.status(200)
                .header("Content-Type", "application/json")
                .body(mock_response_body);
        });

        let api_key = "mock_api_key";
        let instructions = "Analyze the following data:";
        let data_json = r#"{"key": "value"}"#;
        
        let result = analyze_data_with_gpt4(&api_key, instructions, data_json).await;
        
        mock.assert();
        
        assert_eq!(result.unwrap(), "This is a mock response from GPT-4.");
    }
}
