import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import os
from datetime import datetime

class VoiceRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Ses Kaydetme Uygulaması")
        self.root.geometry("400x300")
        
        # Ses kayıt parametreleri
        self.sample_rate = 44100  # Hz
        self.channels = 1  # Mono kayıt
        self.recording = False
        self.audio_data = []
        
        # Kayıt klasörü oluştur
        self.recordings_dir = "recordings"
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Başlık
        title_label = ttk.Label(main_frame, text="Ses Kaydetme Uygulaması", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Kayıt durumu göstergesi
        self.status_label = ttk.Label(main_frame, text="Hazır", 
                                     font=("Arial", 12))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Kayıt butonu
        self.record_button = ttk.Button(main_frame, text="Kayıt Başlat", 
                                       command=self.toggle_recording)
        self.record_button.grid(row=2, column=0, columnspan=2, pady=10, 
                               ipadx=20, ipady=10)
        
        # Oynatma butonu
        self.play_button = ttk.Button(main_frame, text="Son Kaydı Oynat", 
                                     command=self.play_last_recording,
                                     state="disabled")
        self.play_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Dosya seçme butonu
        self.select_button = ttk.Button(main_frame, text="Dosya Seç ve Oynat", 
                                       command=self.select_and_play)
        self.select_button.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Kayıt süresi göstergesi
        self.duration_label = ttk.Label(main_frame, text="")
        self.duration_label.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Kayıtlı dosyalar listesi
        ttk.Label(main_frame, text="Kayıtlı Dosyalar:").grid(row=6, column=0, 
                                                             columnspan=2, pady=(20,5))
        
        # Listbox ve scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.files_listbox = tk.Listbox(list_frame, height=6)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)
        
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        list_frame.columnconfigure(0, weight=1)
        
        # Seçili dosyayı oynat butonu
        ttk.Button(main_frame, text="Seçili Dosyayı Oynat", 
                  command=self.play_selected).grid(row=8, column=0, columnspan=2, pady=5)
        
        # Grid ağırlıkları
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Dosya listesini güncelle
        self.update_file_list()
        
        # Kayıt süresi sayacı
        self.record_start_time = None
        self.duration_thread = None
        
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            self.recording = True
            self.audio_data = []
            self.record_start_time = datetime.now()
            
            # UI güncellemeleri
            self.record_button.config(text="Kayıt Durdur")
            self.status_label.config(text="Kayıt ediliyor...")
            self.play_button.config(state="disabled")
            
            # Süre sayacını başlat
            self.start_duration_counter()
            
            # Kayıt thread'ini başlat
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.start()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Kayıt başlatılamadı: {str(e)}")
            self.recording = False
    
    def _record_audio(self):
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
                    sd.sleep(100)  # 100ms bekle
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Hata", f"Kayıt hatası: {str(e)}"))
    
    def stop_recording(self):
        self.recording = False
        
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        # Ses verisini kaydet
        if self.audio_data:
            audio_array = np.array(self.audio_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kayit_{timestamp}.wav"
            filepath = os.path.join(self.recordings_dir, filename)
            
            try:
                sf.write(filepath, audio_array, self.sample_rate)
                self.last_recording = filepath
                
                # UI güncellemeleri
                self.record_button.config(text="Kayıt Başlat")
                self.status_label.config(text="Kayıt tamamlandı!")
                self.play_button.config(state="normal")
                
                # Süre sayacını durdur
                self.stop_duration_counter()
                
                # Dosya listesini güncelle
                self.update_file_list()
                
                messagebox.showinfo("Başarılı", f"Ses kaydedildi: {filename}")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya kaydedilemedi: {str(e)}")
        else:
            self.record_button.config(text="Kayıt Başlat")
            self.status_label.config(text="Hazır")
            messagebox.showwarning("Uyarı", "Kayıt verisi bulunamadı!")
    
    def start_duration_counter(self):
        def update_duration():
            while self.recording and self.record_start_time:
                elapsed = datetime.now() - self.record_start_time
                seconds = int(elapsed.total_seconds())
                minutes = seconds // 60
                seconds = seconds % 60
                duration_text = f"Süre: {minutes:02d}:{seconds:02d}"
                self.root.after(0, lambda: self.duration_label.config(text=duration_text))
                threading.Event().wait(1)  # 1 saniye bekle
        
        self.duration_thread = threading.Thread(target=update_duration)
        self.duration_thread.start()
    
    def stop_duration_counter(self):
        if self.duration_thread:
            self.duration_thread.join(timeout=1)
        self.duration_label.config(text="")
    
    def play_last_recording(self):
        if hasattr(self, 'last_recording') and os.path.exists(self.last_recording):
            self.play_audio_file(self.last_recording)
        else:
            messagebox.showwarning("Uyarı", "Oynatılacak kayıt bulunamadı!")
    
    def select_and_play(self):
        file_path = filedialog.askopenfilename(
            title="Ses Dosyası Seç",
            filetypes=[("Ses Dosyaları", "*.wav *.mp3 *.flac *.ogg"), 
                      ("Tüm Dosyalar", "*.*")]
        )
        if file_path:
            self.play_audio_file(file_path)
    
    def play_selected(self):
        selection = self.files_listbox.curselection()
        if selection:
            filename = self.files_listbox.get(selection[0])
            filepath = os.path.join(self.recordings_dir, filename)
            self.play_audio_file(filepath)
        else:
            messagebox.showwarning("Uyarı", "Lütfen bir dosya seçin!")
    
    def play_audio_file(self, filepath):
        try:
            # Dosyayı oku
            data, sample_rate = sf.read(filepath)
            
            # Oynatma thread'i
            def play_thread():
                self.root.after(0, lambda: self.status_label.config(text="Oynatılıyor..."))
                sd.play(data, sample_rate)
                sd.wait()  # Oynatma bitene kadar bekle
                self.root.after(0, lambda: self.status_label.config(text="Hazır"))
            
            threading.Thread(target=play_thread).start()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya oynatılamadı: {str(e)}")
    
    def update_file_list(self):
        self.files_listbox.delete(0, tk.END)
        if os.path.exists(self.recordings_dir):
            files = [f for f in os.listdir(self.recordings_dir) 
                    if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            files.sort(reverse=True)  # En yeni dosyalar üstte
            for file in files:
                self.files_listbox.insert(tk.END, file)

def main():
    root = tk.Tk()
    app = VoiceRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()