import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pickle

class VoiceFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_file_path):
        """
        Ses dosyasından çeşitli ses özelliklerini çıkarır
        """
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            # Boş dosya kontrolü
            if len(y) == 0:
                return None, None
                
            features = {}
            
            # 1. MFCC (Mel-frequency cepstral coefficients) - En önemli özellik
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features['mfcc_mean'] = np.mean(mfcc, axis=1)
                features['mfcc_std'] = np.std(mfcc, axis=1)
                features['mfcc_delta_mean'] = np.mean(librosa.feature.delta(mfcc), axis=1)
            except Exception as e:
                print(f"MFCC hatası: {e}")
                features['mfcc_mean'] = np.zeros(13)
                features['mfcc_std'] = np.zeros(13)
                features['mfcc_delta_mean'] = np.zeros(13)
            
            # 2. Spektral özellikler
            try:
                features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
            except Exception as e:
                print(f"Spektral özellik hatası: {e}")
                features['spectral_centroid'] = 0
                features['spectral_rolloff'] = 0
                features['spectral_bandwidth'] = 0
                features['spectral_contrast'] = np.zeros(7)
            
            # 3. Zero crossing rate
            try:
                features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            except Exception as e:
                print(f"ZCR hatası: {e}")
                features['zero_crossing_rate'] = 0
            
            # 4. Chroma özelikleri
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = np.mean(chroma, axis=1)
                features['chroma_std'] = np.std(chroma, axis=1)
            except Exception as e:
                print(f"Chroma hatası: {e}")
                features['chroma_mean'] = np.zeros(12)
                features['chroma_std'] = np.zeros(12)
            
            # 5. RMS (Root Mean Square) - Ses seviyesi
            try:
                features['rms'] = np.mean(librosa.feature.rms(y=y))
            except Exception as e:
                print(f"RMS hatası: {e}")
                features['rms'] = 0
            
            # 6. Temel frekans (F0/Pitch) tahmini - Geliştirilmiş
            try:
                # Yöntem 1: piptrack (mevcut)
                pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, threshold=0.1)
                pitch_values_piptrack = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 50 and pitch < 1000:  # Makul aralık
                        pitch_values_piptrack.append(pitch)
                
                # Yöntem 2: YIN algoritması (daha güvenilir)
                pitch_values_yin = []
                try:
                    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
                    pitch_values_yin = [f for f in f0 if not np.isnan(f) and f > 50]
                except:
                    pass
                
                # En iyi sonucu seç
                if len(pitch_values_yin) > len(pitch_values_piptrack) * 0.5:
                    # YIN daha çok geçerli değer buldu
                    pitch_values = pitch_values_yin
                    pitch_method = "YIN"
                elif len(pitch_values_piptrack) > 0:
                    pitch_values = pitch_values_piptrack
                    pitch_method = "piptrack"
                else:
                    pitch_values = []
                    pitch_method = "none"
                
                if pitch_values:
                    features['pitch_mean'] = np.mean(pitch_values)
                    features['pitch_std'] = np.std(pitch_values)
                    print(f"🔍 DEBUG - Pitch method: {pitch_method}, samples: {len(pitch_values)}")
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
                    print(f"🔍 DEBUG - Pitch detection failed with both methods")
                    
            except Exception as e:
                print(f"Pitch hatası: {e}")
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # 7. Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo, np.ndarray):
                    features['tempo'] = float(tempo[0]) if len(tempo) > 0 and not np.isnan(tempo[0]) else 0
                else:
                    features['tempo'] = float(tempo) if not np.isnan(tempo) else 0
            except Exception as e:
                print(f"Tempo hatası: {e}")
                features['tempo'] = 0
            
            # 8. Mel-scale spektrogram (daha az boyut için)
            try:
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)  # 128 yerine 40
                features['mel_mean'] = np.mean(mel, axis=1)
                features['mel_std'] = np.std(mel, axis=1)
            except Exception as e:
                print(f"Mel spektrogram hatası: {e}")
                features['mel_mean'] = np.zeros(40)
                features['mel_std'] = np.zeros(40)
            
            # Feature vektörünü düzleştir
            feature_vector = self._flatten_features(features)
            
            return feature_vector, features
            
        except Exception as e:
            print(f"Genel özellik çıkarma hatası: {str(e)}")
            return None, None
    
    def _flatten_features(self, features):
        """
        Feature dictionary'sini düz bir vektöre çevirir
        """
        flattened = []
        
        # Tek değerli özellikler
        single_features = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 
                          'zero_crossing_rate', 'rms', 'pitch_mean', 'pitch_std', 'tempo']
        
        for feature in single_features:
            value = features.get(feature, 0)
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    flattened.append(float(value[0]) if hasattr(value[0], '__float__') else 0)
                else:
                    flattened.append(0)
            else:
                flattened.append(float(value) if value is not None else 0)
        
        # Çok boyutlu özellikler - sabit boyutlarda tutmak için
        feature_sizes = {
            'mfcc_mean': 13, 'mfcc_std': 13, 'mfcc_delta_mean': 13,
            'spectral_contrast': 7, 'chroma_mean': 12, 'chroma_std': 12,
            'mel_mean': 40, 'mel_std': 40  # 128'den 40'a düşürüldü
        }
        
        for feature_name, expected_size in feature_sizes.items():
            if feature_name in features and features[feature_name] is not None:
                value = features[feature_name]
                if isinstance(value, np.ndarray):
                    # Boyutu normalize et
                    if len(value) >= expected_size:
                        flattened.extend(value[:expected_size].flatten())
                    else:
                        # Eksik değerleri sıfırla doldur
                        padded = np.zeros(expected_size)
                        padded[:len(value)] = value.flatten()[:len(value)]
                        flattened.extend(padded)
                else:
                    # Skaler değer ise, tüm array'i bu değerle doldur
                    flattened.extend([float(value)] * expected_size)
            else:
                # Eksik feature için sıfır array ekle
                flattened.extend([0.0] * expected_size)
        
        return np.array(flattened, dtype=np.float64)
    
    def analyze_voice_characteristics(self, feature_vector, detailed_features):
        """
        Ses özelliklerini analiz eder ve açıklamaları döner
        """
        analysis = {}
        
        if detailed_features is None:
            return analysis
        
        # Ses tonu analizi - DEBUG bilgileri eklendi
        pitch_mean = detailed_features.get('pitch_mean', 0)
        print(f"🔍 DEBUG - Tespit edilen pitch: {pitch_mean:.2f} Hz")
        
        # Pitch threshold'larını erkek sesi için ayarla
        if pitch_mean > 300:
            analysis['voice_type'] = "Çok yüksek tonlu (çocuk sesi olabilir)"
        elif pitch_mean > 200:
            analysis['voice_type'] = "Yüksek tonlu (kadın sesi olabilir)"
        elif pitch_mean > 150:
            analysis['voice_type'] = "Orta-yüksek tonlu"
        elif pitch_mean > 100:
            analysis['voice_type'] = "Orta tonlu (kadın veya erkek)"
        elif pitch_mean > 70:
            analysis['voice_type'] = "Düşük-orta tonlu (erkek sesi olabilir)"
        elif pitch_mean > 50:
            analysis['voice_type'] = "Düşük tonlu (erkek sesi)"
        elif pitch_mean > 0:
            analysis['voice_type'] = "Çok düşük tonlu (derin erkek sesi)"
        else:
            analysis['voice_type'] = "Ton tespit edilemedi - alternatif yöntem denenecek"
        
        # Eğer pitch tespit edilemezse alternatif yöntemler kullan
        if pitch_mean <= 0:
            # Spektral centroid ile tahmin
            spectral_centroid = detailed_features.get('spectral_centroid', 0)
            if spectral_centroid > 2500:
                analysis['voice_type'] += " (Spektral analiz: yüksek tonlu)"
            elif spectral_centroid > 1500:
                analysis['voice_type'] += " (Spektral analiz: orta tonlu)"
            else:
                analysis['voice_type'] += " (Spektral analiz: düşük tonlu)"
        
        # Konuşma hızı
        tempo = detailed_features.get('tempo', 0)
        print(f"🔍 DEBUG - Tespit edilen tempo: {tempo:.2f} BPM")
        if tempo > 140:
            analysis['speech_rate'] = "Hızlı konuşma"
        elif tempo > 100:
            analysis['speech_rate'] = "Normal hızda konuşma"
        elif tempo > 60:
            analysis['speech_rate'] = "Yavaş konuşma"
        else:
            analysis['speech_rate'] = "Tempo tespit edilemedi"
        
        # Ses kalitesi
        rms = detailed_features.get('rms', 0)
        print(f"🔍 DEBUG - RMS seviye: {rms:.4f}")
        if rms > 0.1:
            analysis['volume'] = "Yüksek ses seviyesi"
        elif rms > 0.05:
            analysis['volume'] = "Normal ses seviyesi"
        elif rms > 0.01:
            analysis['volume'] = "Düşük ses seviyesi"
        else:
            analysis['volume'] = "Çok düşük ses seviyesi"
        
        # Spektral karakteristik
        spectral_centroid = detailed_features.get('spectral_centroid', 0)
        print(f"🔍 DEBUG - Spektral centroid: {spectral_centroid:.2f} Hz")
        if spectral_centroid > 3000:
            analysis['tone_quality'] = "Parlak, net ses"
        elif spectral_centroid > 1500:
            analysis['tone_quality'] = "Dengeli ses tonu"
        elif spectral_centroid > 800:
            analysis['tone_quality'] = "Derin, yumuşak ses"
        else:
            analysis['tone_quality'] = "Çok derin ses"
        
        return analysis
    
    def visualize_features(self, audio_file_path, save_path=None):
        """
        Ses dosyasının özelliklerini görselleştirir
        """
        try:
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Ses Analizi: {os.path.basename(audio_file_path)}', fontsize=16)
            
            # 1. Dalga formu
            axes[0, 0].plot(librosa.times_like(y, sr=sr), y)
            axes[0, 0].set_title('Dalga Formu')
            axes[0, 0].set_xlabel('Zaman (s)')
            axes[0, 0].set_ylabel('Amplitüd')
            
            # 2. MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            im1 = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axes[0, 1])
            axes[0, 1].set_title('MFCC')
            plt.colorbar(im1, ax=axes[0, 1], format='%+2.0f dB')
            
            # 3. Mel-scale spektrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            im2 = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=axes[1, 0])
            axes[1, 0].set_title('Mel Spektrogram')
            plt.colorbar(im2, ax=axes[1, 0], format='%+2.0f dB')
            
            # 4. Spektral özellikler
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            frames = range(len(spectral_centroids))
            t = librosa.frames_to_time(frames, sr=sr)
            axes[1, 1].plot(t, spectral_centroids)
            axes[1, 1].set_title('Spektral Merkez')
            axes[1, 1].set_xlabel('Zaman (s)')
            axes[1, 1].set_ylabel('Hz')
            
            # 5. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            im3 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, ax=axes[2, 0])
            axes[2, 0].set_title('Chroma')
            plt.colorbar(im3, ax=axes[2, 0])
            
            # 6. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            frames = range(len(zcr))
            t = librosa.frames_to_time(frames, sr=sr)
            axes[2, 1].plot(t, zcr)
            axes[2, 1].set_title('Zero Crossing Rate')
            axes[2, 1].set_xlabel('Zaman (s)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Görselleştirme kaydedildi: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Görselleştirme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_features(self, features, filename):
        """
        Özellik vektörlerini dosyaya kaydet
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(features, f)
            print(f"Özellikler kaydedildi: {filename}")
        except Exception as e:
            print(f"Kaydetme hatası: {str(e)}")
    
    def load_features(self, filename):
        """
        Kayıtlı özellik vektörlerini yükle
        """
        try:
            with open(filename, 'rb') as f:
                features = pickle.load(f)
            return features
        except Exception as e:
            print(f"Yükleme hatası: {str(e)}")
            return None

# Test fonksiyonu
def test_feature_extraction():
    """
    Özellik çıkarma modülünü test eder
    """
    print("🎤 Ses Özelliği Çıkarma Testi Başlıyor...")
    extractor = VoiceFeatureExtractor()  # Import yok, direkt class kullan
    
    # Recordings klasöründeki dosyaları test et
    recordings_dir = "recordings"
    if os.path.exists(recordings_dir):
        files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        
        if files:
            print(f"\n📁 Bulunan ses dosyaları: {len(files)}")
            for i, file in enumerate(files):
                print(f"   {i+1}. {file}")
            
            # Tüm dosyaları test et
            for file in files:
                test_file = os.path.join(recordings_dir, file)
                print(f"\n" + "="*60)
                print(f"📁 Test edilen dosya: {file}")
                print("="*60)
                
                # Özellik çıkar
                feature_vector, detailed_features = extractor.extract_features(test_file)
                
                if feature_vector is not None:
                    print(f"✅ Özellik vektörü boyutu: {len(feature_vector)}")
                    
                    # Analiz yap
                    analysis = extractor.analyze_voice_characteristics(feature_vector, detailed_features)
                    print(f"\n🎯 Ses Analizi Sonuçları:")
                    for key, value in analysis.items():
                        print(f"   • {key}: {value}")
                    
                    # Detaylı özellik bilgileri
                    print(f"\n🔍 Ham Değerler:")
                    if detailed_features:
                        print(f"   • Pitch ortalama: {detailed_features.get('pitch_mean', 0):.2f} Hz")
                        print(f"   • Pitch std: {detailed_features.get('pitch_std', 0):.2f} Hz")
                        print(f"   • RMS seviye: {detailed_features.get('rms', 0):.4f}")
                        print(f"   • Spektral merkez: {detailed_features.get('spectral_centroid', 0):.2f} Hz")
                        print(f"   • Spektral rolloff: {detailed_features.get('spectral_rolloff', 0):.2f} Hz")
                        print(f"   • Tempo: {detailed_features.get('tempo', 0):.2f} BPM")
                        print(f"   • Zero crossing rate: {detailed_features.get('zero_crossing_rate', 0):.4f}")
                    
                else:
                    print("❌ Özellik çıkarma başarısız!")
            
            # İlk dosya için görselleştirme
            if files:
                first_file = os.path.join(recordings_dir, files[0])
                print(f"\n📊 İlk dosya için görselleştirme açılıyor: {files[0]}")
                try:
                    extractor.visualize_features(first_file)
                except Exception as e:
                    print(f"Görselleştirme hatası: {e}")
                
        else:
            print("⚠️  Test edilecek ses dosyası bulunamadı!")
            print("   Lütfen recordings/ klasörüne .wav dosyası ekleyin.")
    else:
        print("❌ Recordings klasörü bulunamadı!")
        print("   Önce voice_recorder.py ile ses kayıtları yapın.")

if __name__ == "__main__":
    test_feature_extraction()