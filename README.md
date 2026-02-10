# AImy
AImy is fully offline, vision-enabled AI voice assistant.  It runs on an a Raspberry Pi 5 with the LLM 8850 accelerator from Axera and M5Stack.  

[<img src="https://i.imgur.com/dnMwABX.png" width="800" alt="image of AImy voice assistant dashboard" />](https://i.imgur.com/dnMwABX.png)

[Watch a short demo video here!](https://www.youtube.com/watch?v=sHD0hleZxH4)

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

Vision - Yolo11x - [Axera Yolo11 HF Repo](https://huggingface.co/AXERA-TECH/YOLO11)  
ASR - Sensevoice - [Axera SenseVoice HF Repo](https://huggingface.co/AXERA-TECH/SenseVoice)  
LLM - Qwen2.5-1.5B-IT-int8 - [Axera Qwen2.5-1.5B-IT-int8-python repo](https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct-python)  
TTS - MeloTTS - [Axera MeloTTS HF Repo](https://huggingface.co/AXERA-TECH/MeloTTS)  
Wakeword detection - Vosk - [Vosk model page](https://alphacephei.com/vosk/models)  

## Installation and running

1. Clone repo  
```
git clone https://github.com/malonestar/AImy.git
```  

2. Make installation script executable and run
```
chmod +x install.sh
./install.sh
```
  - The installation script will:
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
aimyenv
```
Please note, you will need to close and open a new terminal first or merge the new bashrc alias with your current session by running:  
```
source ~/.bashrc
```

4. Run
```
python main.py
```


   
