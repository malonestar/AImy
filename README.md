# AImy
AImy is fully offline, vision-enabled AI voice assistant.  It runs on an a Raspberry Pi 5 with the LLM 8850 accelerator from Axera and M5Stack.  

[<img src="/media/06_AIMY_demo_dashboard_20260208.png" width="800" alt="image of AImy voice assistant dashboard" />](/media/06_AIMY_demo_dashboard_20260208.png)

[Watch a short demo video here!](https://youtu.be/sHD0hleZxH4)

## Prerequisites  
- LLM 8850 Software Installation - [Official M5Stack Documentation](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install)

## Hardware
- Raspberry Pi 5 (8gb)
- [M5Stack LLM 8850](https://shop.m5stack.com/products/ai-8850-llm-accleration-m-2-module-ax8850?srsltid=AfmBOorb8a6pqowOzc3W6XTMt_rQi0eRfS348f9Q4nSUv8zWs7-WOU4Q&variant=46766221459713)
- Raspberry Pi M.2 Hat+
- Raspberry Pi Camera Module 3
- USB Speaker, USB Microphone (I used [Waveshare USB to Audio card](https://www.waveshare.com/usb-to-audio.htm?srsltid=AfmBOoq6BNzQbLJWfXqs6gvRb73WYQn_J39oWneCJ5zVkfmi9arjc4iu))

The LLM 8850 M.2 Card is also available in a kit from M5Stack that includes their own version of an M.2 hat - [see production information here](https://shop.m5stack.com/products/ai-8850-llm-accelerator-m-2-kit-8gb-version-ax8850?srsltid=AfmBOooJJABFyCjqG_JYt5sdAtXISGdD7OZnD173HBugBlfWJCz8MYIi).  This version would not require the Raspberry Pi M.2 Hat+ listed above.  

## Models
The code that does the heavy lifting in this project was authored by [Axera Tech](https://huggingface.co/AXERA-TECH).  I started with examples in the HF repo and expanded and adapted them to use additional hardware like the camera and microphone.  Models used and their specific page within the Axera HF repo are below.  

Vision - Yolo11x - [Axera Yolo11 HF Repo](https://huggingface.co/AXERA-TECH/YOLO11)  
ASR - Sensevoice - [Axera SenseVoice HF Repo](https://huggingface.co/AXERA-TECH/SenseVoice)  
LLM - Qwen2.5-1.5B-IT-int8 - [Axera Qwen2.5-1.5B-IT-int8-python repo](https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct-python)  
TTS - MeloTTS - [Axera MeloTTS HF Repo](https://huggingface.co/AXERA-TECH/MeloTTS)  
Wakeword detection - Vosk - [Vosk model page](https://alphacephei.com/vosk/models)  
Wakeword detection - Porcupine / Picovoice - [Picovoice](https://picovoice.ai/)  
  - I have included a copy of my "hey amy" porcupine model in the repo

## Installation and running  

Installation takes about 5-10 mins total, including model downloads.  
I have a [short video](https://youtu.be/xxNVnvPNR0Y) showing the full installation process, up through running your first voice prompt.  Took about 8 minutes total! 

1. Clone repo  
```
git clone https://github.com/malonestar/AImy.git
```  

2. Make installation script executable and run
```
chmod +x install.sh
./install.sh
```
The installation script will:
  - Check system dependencies and install if missing - PiCamera2, OpenCV
  - Create virtual environment 'aimy_venv'
  - Install dependencies from requirements.txt
  - Create bashrc alias to activate venv from anywhere by typing 'aimyenv'
  - Download models via scripts/download_models.py - Qwen and Yolo from Axera HF repos, Sensevoice+MeloTTS+Vosk from Google Drive to maintain compatibility with an older version of the Sensevoice script from Axera


3. Activate virtual environment  
```
source aimy_venv/bin/activate
```
OR  
```
source ~/.bashrc
aimyenv
```
Please note, you will only need to run the 'source ~/.bashrc' one time in the current terminal window, OR you can just close out and open a new terminal.  

4. Run
```
python main.py
```

## Example setups  

[<img src="/media/01_AImy_example_setup.jpg" width="600" alt="AImy example setup" />](/media/01_AImy_example_setup.jpg)  

[<img src="/media/02_AImy_example_setup_pironman5.jpg" width="600" alt="AImy example setup with Pironman5 Max case" />](/media/02_AImy_example_setup_pironman5.jpg)  







   
