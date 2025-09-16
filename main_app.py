import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from speaker_database import SpeakerIdentificationSystem

class VoiceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Ses Tanıma Sistemi")
        self.root.geometry("600x700")
        
        # Ses sistemi
        self.speaker_system = SpeakerIdentificationSystem()
        
        # Ses kayıt parametreleri
        self.sample_rate = 44100
        self.channels = 1
        self.recording = False
        self.audio_data = []
        
        # Ana arayüzü oluştur
        self.setup_ui()
        
        # Başlangıçta verileri yükle
        self.refresh_data()
        
    def setup_ui(self):
        # Ana container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Başlık
        title_label = ttk.Label(main_frame, text="🎤 Ses Tanıma Sistemi", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=20)
        
        # === KİŞİ YÖNETİMİ BÖLÜMÜ ===
        person_frame = ttk.LabelFrame(main_frame, text="👤 Kişi Yönetimi", padding="10")
        person_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Yeni kişi ekleme
        ttk.Label(person_frame, text="Yeni Kişi Adı:").grid(row=0, column=0, sticky=tk.W)
        self.person_name_var = tk.StringVar()
        ttk.Entry(person_frame, textvariable=self.person_name_var, width=20).grid(row=0, column=1, padx=5)
        ttk.Button(person_frame, text="Kişi Ekle", command=self.add_person).grid(row=0, column=2, padx=5)

        # Kişi silme
        ttk.Label(person_frame, text="Kişi Sil:").grid(row=1, column=0, sticky=tk.W)
        self.delete_person_var = tk.StringVar()
        self.delete_person_combo = ttk.Combobox(person_frame, textvariable=self.delete_person_var, width=15, state="readonly")
        self.delete_person_combo.grid(row=1, column=1, padx=5)
        ttk.Button(person_frame, text="Kişi Sil", command=self.delete_person).grid(row=1, column=2, padx=5)
        
        # === SES KAYDETME BÖLÜMÜ ===
        record_frame = ttk.LabelFrame(main_frame, text="🎤 Ses Kaydetme", padding="10")
        record_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Kişi seçimi
        ttk.Label(record_frame, text="Kişi Seç:").grid(row=0, column=0, sticky=tk.W)
        self.selected_person_var = tk.StringVar()
        self.person_combo = ttk.Combobox(record_frame, textvariable=self.selected_person_var, 
                                        width=15, state="readonly")
        self.person_combo.grid(row=0, column=1, padx=5)
        
        # Kayıt kontrolleri
        self.record_button = ttk.Button(record_frame, text="🔴 Kayıt Başlat", 
                                       command=self.toggle_recording)
        self.record_button.grid(row=0, column=2, padx=5)
        
        # Kayıt durumu
        self.record_status = ttk.Label(record_frame, text="Hazır", font=("Arial", 10))
        self.record_status.grid(row=1, column=0, columnspan=3, pady=5)
        
        # === MODEL EĞİTİMİ BÖLÜMÜ ===
        model_frame = ttk.LabelFrame(main_frame, text="🤖 Model Eğitimi", padding="10")
        model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(model_frame, text="🧠 Model Eğit", command=self.train_model).grid(row=0, column=0, padx=5)
        ttk.Button(model_frame, text="📂 Model Yükle", command=self.load_model).grid(row=0, column=1, padx=5)
        
        self.model_status = ttk.Label(model_frame, text="Model durumu: Belirsiz", font=("Arial", 10))
        self.model_status.grid(row=1, column=0, columnspan=2, pady=5)
        
        # === SES TANIMA BÖLÜMÜ ===
        identify_frame = ttk.LabelFrame(main_frame, text="🔍 Ses Tanıma", padding="10")
        identify_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(identify_frame, text="📁 Dosya Seç ve Tanı", 
                  command=self.identify_from_file).grid(row=0, column=0, padx=5)
        ttk.Button(identify_frame, text="🎤 Canlı Tanıma", 
                  command=self.live_identification).grid(row=0, column=1, padx=5)
        
        # Tanıma sonucu
        self.result_label = ttk.Label(identify_frame, text="Sonuç: -", 
                                     font=("Arial", 12, "bold"))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # === İSTATİSTİKLER BÖLÜMÜ ===
        stats_frame = ttk.LabelFrame(main_frame, text="📊 İstatistikler", padding="10")
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Kayıtlı kişiler listesi
        self.stats_text = tk.Text(stats_frame, height=8, width=60)
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Yenile butonu
        ttk.Button(stats_frame, text="🔄 Yenile", command=self.refresh_data).grid(row=1, column=0, pady=5)
        
        # Grid konfigürasyonu
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
    
    def add_person(self):
        """Yeni kişi ekle"""
        name = self.person_name_var.get().strip()
        if not name:
            messagebox.showwarning("Uyarı", "Lütfen kişi adını girin!")
            return
        
        try:
            self.speaker_system.add_speaker(name)
            self.person_name_var.set("")
            self.refresh_data()
            messagebox.showinfo("Başarılı", f"{name} eklendi!")
        except Exception as e:
            messagebox.showerror("Hata", f"Kişi eklenemedi: {str(e)}")
    
    def toggle_recording(self):
        """Kayıt başlat/durdur"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Ses kaydını başlat"""
        person_name = self.selected_person_var.get()
        if not person_name:
            messagebox.showwarning("Uyarı", "Lütfen bir kişi seçin!")
            return
        
        try:
            self.recording = True
            self.audio_data = []
            
            # UI güncellemeleri
            self.record_button.config(text="🛑 Kayıt Durdur")
            self.record_status.config(text=f"{person_name} için kayıt ediliyor...")
            
            # Kayıt thread'ini başlat
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.start()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Kayıt başlatılamadı: {str(e)}")
            self.recording = False
    
    def _record_audio(self):
        """Ses kayıt callback'i"""
        def callback(indata, frames, time, status):
            if status:
                print(f"Kayıt hatası: {status}")
            if self.recording:
                self.audio_data.extend(indata.copy())
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                              channels=self.channels, 
                              callback=callback):
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Hata", f"Kayıt hatası: {str(e)}"))
    
    def stop_recording(self):
        """Ses kaydını durdur"""
        self.recording = False
        
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        person_name = self.selected_person_var.get()
        
        # Ses verisini kaydet
        if self.audio_data:
            try:
                audio_array = np.array(self.audio_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"temp_recording_{timestamp}.wav"
                temp_filepath = os.path.join("recordings", temp_filename)
                
                # recordings klasörü yoksa oluştur
                os.makedirs("recordings", exist_ok=True)
                
                # Geçici dosyaya kaydet
                sf.write(temp_filepath, audio_array, self.sample_rate)
                
                # Speaker sistemine ekle
                if self.speaker_system.add_voice_sample(person_name, temp_filepath):
                    # UI güncellemeleri
                    self.record_button.config(text="🔴 Kayıt Başlat")
                    self.record_status.config(text="Kayıt başarıyla eklendi!")
                    self.refresh_data()
                    messagebox.showinfo("Başarılı", f"{person_name} için ses örneği eklendi!")
                else:
                    messagebox.showerror("Hata", "Ses analizi başarısız!")
                
                # Geçici dosyayı sil
                try:
                    os.remove(temp_filepath)
                except:
                    pass
                    
            except Exception as e:
                messagebox.showerror("Hata", f"Ses kaydedilemedi: {str(e)}")
        else:
            messagebox.showwarning("Uyarı", "Kayıt verisi bulunamadı!")
        
        # UI'yi sıfırla
        self.record_button.config(text="🔴 Kayıt Başlat")
        self.record_status.config(text="Hazır")
    
    def train_model(self):
        """Model eğitimi yap"""
        def train_thread():
            try:
                self.root.after(0, lambda: self.model_status.config(text="Model eğitiliyor..."))
                success = self.speaker_system.train_model()
                
                if success:
                    self.root.after(0, lambda: [
                        self.model_status.config(text="Model eğitimi başarılı!"),
                        messagebox.showinfo("Başarılı", "Model eğitimi tamamlandı!"),
                        self.refresh_data()
                    ])
                else:
                    self.root.after(0, lambda: [
                        self.model_status.config(text="Model eğitimi başarısız!"),
                        messagebox.showerror("Hata", "Model eğitimi başarısız!")
                    ])
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Hata", f"Eğitim hatası: {str(e)}"))
        
        threading.Thread(target=train_thread).start()
    
    def load_model(self):
        """Kayıtlı modeli yükle"""
        if self.speaker_system.load_model():
            self.model_status.config(text="Model başarıyla yüklendi!")
            messagebox.showinfo("Başarılı", "Model yüklendi!")
            self.refresh_data()
        else:
            self.model_status.config(text="Model yüklenemedi!")
            messagebox.showwarning("Uyarı", "Eğitilmiş model bulunamadı!")
    
    def identify_from_file(self):
        """Dosyadan ses tanıma"""
        if not self.speaker_system.is_trained:
            if not self.speaker_system.load_model():
                messagebox.showwarning("Uyarı", "Önce model eğitin veya yükleyin!")
                return
        file_path = filedialog.askopenfilename(
            title="Tanınacak Ses Dosyasını Seçin",
            filetypes=[("Ses Dosyaları", "*.wav *.mp3 *.flac *.ogg"),
                      ("Tüm Dosyalar", "*.*")]
        )
        if file_path:
            try:
                speaker_name = self.speaker_system.identify_speaker(file_path)
                if speaker_name and speaker_name != "Bilinmeyen_Kisi":
                    result_text = f"🎯 {speaker_name}"
                    self.result_label.config(text=result_text, foreground="green")
                else:
                    result_text = f"❓ Bilinmeyen kişi"
                    self.result_label.config(text=result_text, foreground="orange")
                messagebox.showinfo("Tanıma Sonucu", result_text)
            except Exception as e:
                messagebox.showerror("Hata", f"Tanıma hatası: {str(e)}")
    
    def live_identification(self):
        """Canlı ses tanıma"""
        if not self.speaker_system.is_trained:
            if not self.speaker_system.load_model():
                messagebox.showwarning("Uyarı", "Önce model eğitin veya yükleyin!")
                return
        # Basit canlı tanıma - 3 saniyelik kayıt
        result = messagebox.askyesno("Canlı Tanıma", 
                                   "3 saniye konuşacaksınız. Hazır mısınız?")
        if not result:
            return
        try:
            # 3 saniye kayıt yap
            self.result_label.config(text="🎤 Dinleniyor... (3 saniye)", foreground="blue")
            self.root.update()
            duration = 3  # saniye
            recording = sd.rec(int(duration * self.sample_rate), 
                             samplerate=self.sample_rate, 
                             channels=self.channels)
            sd.wait()  # Kayıt bitene kadar bekle
            # Geçici dosyaya kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_live_{timestamp}.wav"
            sf.write(temp_file, recording, self.sample_rate)
            # Tanıma yap
            speaker_name = self.speaker_system.identify_speaker(temp_file)
            if speaker_name and speaker_name != "Bilinmeyen_Kisi":
                result_text = f"🎯 {speaker_name}"
                self.result_label.config(text=result_text, foreground="green")
            else:
                result_text = f"❓ Bilinmeyen kişi"
                self.result_label.config(text=result_text, foreground="orange")
            # Geçici dosyayı sil
            try:
                os.remove(temp_file)
            except:
                pass
            messagebox.showinfo("Canlı Tanıma Sonucu", result_text)
        except Exception as e:
            messagebox.showerror("Hata", f"Canlı tanıma hatası: {str(e)}")
            self.result_label.config(text="Sonuç: -", foreground="black")
    
    def delete_person(self):
        """Kişi ve verilerini sil"""
        person_name = self.delete_person_var.get()
        if not person_name:
            messagebox.showwarning("Uyarı", "Lütfen silinecek kişiyi seçin!")
            return
        result = messagebox.askyesno("Kişi Sil", f"{person_name} ve tüm verileri silinsin mi?")
        if not result:
            return
        try:
            success = self.speaker_system.delete_speaker(person_name)
            if success:
                messagebox.showinfo("Başarılı", f"{person_name} ve verileri silindi!")
                self.refresh_data()
            else:
                messagebox.showerror("Hata", f"Kişi silinemedi: {person_name}")
        except Exception as e:
            messagebox.showerror("Hata", f"Silme hatası: {str(e)}")

    def refresh_data(self):
        """Verileri yenile"""
        try:
            # İstatistikleri al
            stats = self.speaker_system.get_statistics()
            
            # Kişi listesini güncelle
            speaker_names = [speaker["name"] for speaker in stats["speakers"]]
            self.person_combo["values"] = speaker_names
            self.delete_person_combo["values"] = speaker_names
            
            # İstatistik metnini güncelle
            self.stats_text.delete(1.0, tk.END)
            
            stats_info = f"📊 SİSTEM İSTATİSTİKLERİ\n"
            stats_info += f"{'='*40}\n"
            stats_info += f"Toplam Kişi: {stats['total_speakers']}\n"
            stats_info += f"Toplam Ses Örneği: {stats['total_samples']}\n"
            stats_info += f"Model Durumu: {'Eğitildi' if stats['model_trained'] else 'Eğitilmedi'}\n\n"
            
            if "model_info" in stats:
                model_info = stats["model_info"]
                stats_info += f"🤖 MODEL BİLGİLERİ\n"
                stats_info += f"Tip: {model_info.get('model_type', 'Bilinmiyor')}\n"
                stats_info += f"Doğruluk: {model_info.get('accuracy', 0):.3f}\n"
                stats_info += f"Eğitim Örneği: {model_info.get('training_samples', 0)}\n\n"
            
            stats_info += f"👥 KAYITLI KİŞİLER\n"
            stats_info += f"{'-'*40}\n"
            
            if stats["speakers"]:
                for speaker in stats["speakers"]:
                    stats_info += f"• {speaker['name']}: {speaker['sample_count']} örnek\n"
            else:
                stats_info += "Henüz kimse eklenmemiş.\n"
            
            self.stats_text.insert(1.0, stats_info)
            
            # Model durumunu güncelle
            if stats['model_trained']:
                self.model_status.config(text="Model durumu: Eğitildi ✅")
            else:
                self.model_status.config(text="Model durumu: Eğitilmedi ❌")
                
        except Exception as e:
            print(f"Veri yenileme hatası: {e}")

def main():
    root = tk.Tk()
    app = VoiceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()