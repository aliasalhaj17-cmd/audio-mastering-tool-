import os
from df.enhance import enhance, load_audio, save_audio, init_df
from pedalboard import Pedalboard, HighpassFilter, Compressor, HighShelfFilter, Limiter
import soundfile as sf

def process_audio(input_file_path, output_file_path):
    # ---- 1. مرحلة التنقية الذكية (AI Noise Reduction) ----
    # نموذج DeepFilterNet يقوم بإزالة الضوضاء والصدى تلقائياً
    model, df_state = init_df()
    audio, audio_sr = load_audio(input_file_path, sr=df_state.sr())
    enhanced_audio = enhance(model, df_state, audio)
    
    # حفظ ملف مؤقت نقي لبدء الهندسة عليه
    temp_clean_path = "temp_clean.wav"
    save_audio(temp_clean_path, enhanced_audio, audio_sr)

    # ---- 2. مرحلة الهندسة والماسترنج (DSP Mastering) ----
    # قراءة الصوت النقي
    audio_data, sample_rate = sf.read(temp_clean_path)
    
    # بناء لوحة التأثيرات (Effect Rack)
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),         # قص الترددات المنخفضة المزعجة
        Compressor(threshold_db=-16, ratio=3),          # موازنة ديناميكية الصوت
        HighShelfFilter(cutoff_frequency_hz=5000, gain_db=2), # إضافة لمعان ووضوح للصوت (Air)
        Limiter(threshold_db=-1.0, release_ms=100)      # رفع الصوت لأقصى حد احترافي بدون تشويه
    ])
    
    # تطبيق التأثيرات
    effected_audio = board(audio_data, sample_rate)
    
    # ---- 3. تصدير الملف النهائي ----
    sf.write(output_file_path, effected_audio, sample_rate)
    
    # تنظيف الملفات المؤقتة
    if os.path.exists(temp_clean_path):
        os.remove(temp_clean_path)

# تشغيل الأداة
process_audio("my_voice.wav", "perfect_voice.wav")
