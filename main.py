import os
import soundfile as sf
from df.enhance import enhance, load_audio, save_audio, init_df
from pedalboard import Pedalboard, HighpassFilter, Compressor, HighShelfFilter, Limiter
import gradio as gr

# 1. الدالة الأساسية لمعالجة الصوت (تنقية + هندسة + ماسترنج)
def process_audio(input_file_path, output_file_path):
    # ---- مرحلة التنقية الذكية (AI Noise Reduction) ----
    # استدعاء نموذج DeepFilterNet لتنظيف الضوضاء والصدى
    model, df_state = init_df()
    audio, audio_sr = load_audio(input_file_path, sr=df_state.sr())
    enhanced_audio = enhance(model, df_state, audio)
    
    # حفظ ملف مؤقت نقي لبدء الماسترنج عليه
    temp_clean_path = "temp_clean.wav"
    save_audio(temp_clean_path, enhanced_audio, audio_sr)

    # ---- مرحلة الهندسة والماسترنج (DSP Mastering) ----
    # قراءة الصوت النقي الذي تم تنظيفه
    audio_data, sample_rate = sf.read(temp_clean_path)
    
    # بناء فلاتر الهندسة الصوتية الاحترافية
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),         # إزالة الترددات المنخفضة المزعجة (الهواء والاهتزازات)
        Compressor(threshold_db=-16, ratio=3),          # موازنة علو وانخفاض الصوت وجعله متناسقاً
        HighShelfFilter(cutoff_frequency_hz=5000, gain_db=2), # إضافة لمعان ووضوح لصوت الإنسان (Presence)
        Limiter(threshold_db=-1.0, release_ms=100)      # رفع الصوت لأعًلى درجة احترافية آمنة دون تشويه
    ])
    
    # تطبيق الفلاتر على الصوت
    effected_audio = board(audio_data, sample_rate)
    
    # ---- حفظ وتصدير الملف النهائي ----
    sf.write(output_file_path, effected_audio, sample_rate)
    
    # مسح الملف المؤقت للحفاظ على مساحة الجهاز
    if os.path.exists(temp_clean_path):
        os.remove(temp_clean_path)

# 2. بناء واجهة الويب (Gradio Interface)
def web_interface(audio_file):
    if audio_file is None:
        return None
    output_path = "processed_audio.wav"
    process_audio(audio_file, output_path)
    return output_path

# تشغيل الواجهة الرسومية للمستخدم
demo = gr.Interface(
    fn=web_interface,
    inputs=gr.Audio(type="filepath", label="ارفع الملف الصوتي المراد تنقيته وهندسته"),
    outputs=gr.Audio(label="الملف الصوتي النهائي الاحترافي جاهز للتحميل"),
    title="🎙️ أداة الهندسة الصوتية والماسترنج الذكي"
)

# تشغيل الأداة
if __name__ == "__main__":
    demo.launch()
