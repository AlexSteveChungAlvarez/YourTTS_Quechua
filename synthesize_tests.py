from TTS.api import TTS
import os

if __name__ == '__main__':
    os.makedirs("audios", exist_ok=True)
    text1 = input("\nReference text 1\n")
    text2 = input("\nReference text 2\n")
    name = input("\nReference audio .wav or Enter\n")
    lang = "qu" 
    
    command=f"tts --text \"{text1}\" --model_path recipes\\vctk\yourtts\YourTTS-QU-SINGLE-January-13-2023_12+38AM-14d45b53\\best_model.pth --config_path recipes\\vctk\yourtts\YourTTS-QU-SINGLE-January-13-2023_12+38AM-14d45b53\config.json --out_path yourTTS\{name}_test2_quechua_only.wav --speaker_wav C:\\Users\\Aorus\\Documents\\Voice_Cloning_Quechua\\{name}.wav --language_idx {lang}"
    os.system("cmd /c "+command+"")

    command=f"tts --text \"{text2}\" --model_path recipes\\vctk\yourtts\YourTTS-finetune-QU-LibriTTS-January-17-2023_06+51PM-14d45b53\\best_model.pth --config_path recipes\\vctk\yourtts\YourTTS-finetune-QU-LibriTTS-January-17-2023_06+51PM-14d45b53\config.json --out_path yourTTS\{name}_test2_quechua.wav --speaker_wav C:\\Users\\Aorus\\Documents\\Voice_Cloning_Quechua\\{name}.wav --language_idx {lang}"
    os.system("cmd /c "+command+"")
