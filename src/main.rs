use eframe::egui;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::runtime::Builder;
use std::fs;
use dotenvy::dotenv;
use std::env;

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
    response: String,
    my_message: String,
    api_key: String,
    data_json: String,
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
            response: String::new(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 한글 폰트를 로드
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

                let runtime = Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();

                match runtime.block_on(analyze_data_with_gpt4(
                    &api_key,
                    &my_message,
                    &data_json,
                )) {
                    Ok(advice) => self.response = advice,
                    Err(e) => self.response = format!("Error: {}", e),
                }
            }
            
            ui.label("딜러의 대답 :");
            ui.add(
                egui::TextEdit::multiline(&mut self.response)
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
