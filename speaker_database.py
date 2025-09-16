import os
import json
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
from voice_features import VoiceFeatureExtractor
from datetime import datetime

class SpeakerIdentificationSystem:
    def __init__(self, database_dir="voice_database"):
        self.database_dir = database_dir
        self.speakers_dir = os.path.join(database_dir, "speakers")
        self.models_dir = os.path.join(database_dir, "models")
        self.metadata_file = os.path.join(database_dir, "metadata.json")
        
        # Klasörleri oluştur
        os.makedirs(self.speakers_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Voice feature extractor
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Metadata yükle veya oluştur
        self.metadata = self.load_metadata()
        
        # ML modelleri
        self.scaler = StandardScaler()
        self.classifier = None
        self.is_trained = False
        
    def load_metadata(self):
        """Metadata dosyasını yükle"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default metadata
        return {
            "speakers": {},
            "total_samples": 0,
            "last_updated": datetime.now().isoformat(),
            "model_trained": False
        }
    
    def save_metadata(self):
        """Metadata dosyasını kaydet"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def add_speaker(self, speaker_name):
        """Yeni kişi ekle"""
        speaker_id = speaker_name.lower().replace(" ", "_").replace("ç", "c").replace("ğ", "g").replace("ı", "i").replace("ö", "o").replace("ş", "s").replace("ü", "u")
        speaker_dir = os.path.join(self.speakers_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Metadata güncelle
        if speaker_id not in self.metadata["speakers"]:
            self.metadata["speakers"][speaker_id] = {
                "name": speaker_name,
                "speaker_id": speaker_id,
                "samples": [],
                "sample_count": 0,
                "added_date": datetime.now().isoformat()
            }
            self.save_metadata()
            print(f"✅ Yeni kişi eklendi: {speaker_name} (ID: {speaker_id})")
        else:
            print(f"⚠️ Bu kişi zaten mevcut: {speaker_name}")
        
        return speaker_id
    
    def add_voice_sample(self, speaker_name, audio_file_path):
        """Kişi için ses örneği ekle"""
        # Kişiyi ekle (zaten varsa hiçbir şey yapmaz)
        speaker_id = self.add_speaker(speaker_name)
        
        # Ses özelliklerini çıkar
        print(f"🔍 {speaker_name} için ses analiz ediliyor...")
        feature_vector, detailed_features = self.feature_extractor.extract_features(audio_file_path)
        
        if feature_vector is not None:
            # Ses dosyasını kişi klasörüne kopyala
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{speaker_id}_{timestamp}.wav"
            new_file_path = os.path.join(self.speakers_dir, speaker_id, new_filename)
            shutil.copy2(audio_file_path, new_file_path)
            
            # Feature'ları kaydet
            feature_file = os.path.join(self.speakers_dir, speaker_id, f"{speaker_id}_{timestamp}.pkl")
            with open(feature_file, 'wb') as f:
                pickle.dump({
                    'features': feature_vector,
                    'detailed_features': detailed_features,
                    'original_file': audio_file_path,
                    'timestamp': timestamp
                }, f)
            
            # Metadata güncelle
            self.metadata["speakers"][speaker_id]["samples"].append({
                "audio_file": new_filename,
                "feature_file": f"{speaker_id}_{timestamp}.pkl",
                "timestamp": timestamp
            })
            self.metadata["speakers"][speaker_id]["sample_count"] += 1
            self.metadata["total_samples"] += 1
            self.metadata["model_trained"] = False  # Model yeniden eğitilmeli
            
            self.save_metadata()
            print(f"✅ {speaker_name} için ses örneği eklendi (Toplam: {self.metadata['speakers'][speaker_id]['sample_count']})")
            return True
        else:
            print(f"❌ Ses analizi başarısız: {audio_file_path}")
            return False
    
    def get_all_features_and_labels(self):
        """Tüm ses örneklerinin özelliklerini ve etiketlerini getir"""
        features = []
        labels = []
        
        for speaker_id, speaker_data in self.metadata["speakers"].items():
            speaker_dir = os.path.join(self.speakers_dir, speaker_id)
            
            for sample in speaker_data["samples"]:
                feature_file = os.path.join(speaker_dir, sample["feature_file"])
                if os.path.exists(feature_file):
                    try:
                        with open(feature_file, 'rb') as f:
                            data = pickle.load(f)
                            features.append(data['features'])
                            labels.append(speaker_id)
                    except Exception as e:
                        print(f"⚠️ Feature dosyası okunamadı: {feature_file} - {e}")
        
        return np.array(features), np.array(labels)
    
    def train_model(self):
        """Tanıma modelini eğit (geliştirilmiş)"""
        print("🤖 Model eğitimi başlıyor...")
        
        # Tüm özellik ve etiketleri al
        features, labels = self.get_all_features_and_labels()
        
        if len(features) == 0:
            print("❌ Eğitim için veri bulunamadı!")
            return False
        
        if len(np.unique(labels)) < 2:
            print("❌ En az 2 farklı kişinin sesi gerekli!")
            return False
        
        print(f"📊 Eğitim verisi: {len(features)} örnek, {len(np.unique(labels))} kişi")
        print(f"📊 Kişiler: {list(np.unique(labels))}")
        
        # Class balancing
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = {c: w for c, w in zip(np.unique(labels), class_weights)}
        
        # Veri artırma (noise injection)
        # ... (isteğe bağlı olarak eklenebilir)
        
        # Verileri normalize et
        features_scaled = self.scaler.fit_transform(features)
        
        # Birden fazla model dene
        models = {
            'SVM': SVC(kernel='rbf', class_weight=class_weight_dict, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
        }
        
        # XGBoost
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
        except ImportError:
            print("XGBoost yüklü değil, atlanıyor.")
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['LightGBM'] = LGBMClassifier(n_estimators=100)
        except ImportError:
            print("LightGBM yüklü değil, atlanıyor.")
        
        best_model = None
        best_score = 0
        best_model_name = ""
        
        for model_name, model in models.items():
            try:
                # Cross-validation ile performans değerlendir
                if len(features) >= 5:  # Yeterli veri varsa cross-validation yap
                    scores = cross_val_score(model, features_scaled, labels, cv=min(3, len(features)))
                    avg_score = np.mean(scores)
                    print(f"📈 {model_name} CV skoru: {avg_score:.3f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model_name = model_name
                else:
                    # Az veri varsa direkt eğit
                    model.fit(features_scaled, labels)
                    best_model = model
                    best_model_name = model_name
                    break
                    
            except Exception as e:
                print(f"❌ {model_name} eğitimi başarısız: {e}")
        
        if best_model is None:
            print("❌ Hiçbir model eğitilemedi!")
            return False
        
        # En iyi modeli eğit
        self.classifier = best_model
        self.classifier.fit(features_scaled, labels)
        
        # Modeli kaydet
        model_file = os.path.join(self.models_dir, "speaker_model.pkl")
        scaler_file = os.path.join(self.models_dir, "scaler.pkl")
        
        joblib.dump(self.classifier, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        # Metadata güncelle
        self.metadata["model_trained"] = True
        self.metadata["model_info"] = {
            "model_type": best_model_name,
            "accuracy": float(best_score),
            "training_samples": len(features),
            "speakers": list(np.unique(labels)),
            "trained_date": datetime.now().isoformat()
        }
        self.save_metadata()
        
        self.is_trained = True
        print(f"✅ Model eğitimi tamamlandı! ({best_model_name}, Skor: {best_score:.3f})")
        return True
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        model_file = os.path.join(self.models_dir, "speaker_model.pkl")
        scaler_file = os.path.join(self.models_dir, "scaler.pkl")
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                self.classifier = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                print("✅ Model başarıyla yüklendi!")
                return True
            except Exception as e:
                print(f"❌ Model yüklenemedi: {e}")
        else:
            print("⚠️ Eğitilmiş model bulunamadı!")
        
        return False
    
    def identify_speaker(self, audio_file_path):
        """Ses dosyasının kime ait olduğunu tahmin et"""
        if not self.is_trained:
            if not self.load_model():
                print("❌ Eğitilmiş model bulunamadı! Önce model eğitimi yapın.")
                return None
        # Ses özelliklerini çıkar
        feature_vector, detailed_features = self.feature_extractor.extract_features(audio_file_path)
        if feature_vector is None:
            print("❌ Ses analizi başarısız!")
            return None
        # Özellik vektörünü normalize et
        feature_scaled = self.scaler.transform([feature_vector])
        # Tahmin yap
        prediction = self.classifier.predict(feature_scaled)[0]
        # Speaker ID'yi isime çevir
        speaker_name = self.metadata["speakers"].get(prediction, {}).get("name", "Bilinmeyen_Kisi")
        return speaker_name
    
    def add_multiple_samples_from_file(self, speaker_name, audio_file_path, segment_length=3):
        """Bir ses dosyasından birden fazla örnek oluştur"""
        import librosa
        import soundfile as sf
        import tempfile
        import time
        
        # Ses dosyasını yükle
        y, sr = librosa.load(audio_file_path, sr=44100)
        
        # Segment uzunluğunu sample'a çevir
        segment_samples = segment_length * sr
        
        # Dosyayı segmentlere böl
        segments = []
        for i in range(0, len(y) - segment_samples, segment_samples // 2):  # %50 overlap
            segment = y[i:i + segment_samples]
            if len(segment) >= segment_samples:
                segments.append(segment)
        
        if len(segments) == 0:
            # Çok kısa dosya, direkt ekle
            return self.add_voice_sample(speaker_name, audio_file_path)
        
        # Her segment için geçici dosya oluştur ve ekle
        success_count = 0
        temp_files = []  # Silinecek dosyaları takip et
        
        try:
            for i, segment in enumerate(segments):
                # Geçici dosya oluştur
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_files.append(temp_file.name)
                
                # Segment'i dosyaya yaz
                sf.write(temp_file.name, segment, sr)
                temp_file.close()  # Dosyayı kapat
                
                # Ses örneği ekle
                if self.add_voice_sample(speaker_name, temp_file.name):
                    success_count += 1
        
        finally:
            # Tüm geçici dosyaları temizle
            for temp_path in temp_files:
                try:
                    time.sleep(0.1)  # Windows için kısa bekleme
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except PermissionError:
                    # Windows'ta bazen hemen silinemiyor, tekrar dene
                    try:
                        time.sleep(0.5)
                        os.unlink(temp_path)
                    except:
                        print(f"⚠️ Geçici dosya silinemedi: {temp_path}")
                except Exception as e:
                    print(f"⚠️ Dosya silme hatası: {e}")
        
        print(f"✅ {speaker_name} için {success_count} segment eklendi (Toplam {len(segments)} segment)")
        return success_count > 0
    
    def get_speaker_list(self):
        """Kayıtlı kişilerin listesini getir"""
        speakers = []
        for speaker_id, data in self.metadata["speakers"].items():
            speakers.append({
                "name": data["name"],
                "id": speaker_id,
                "sample_count": data["sample_count"]
            })
        return speakers
    
    def get_statistics(self):
        """Sistem istatistiklerini getir"""
        stats = {
            "total_speakers": len(self.metadata["speakers"]),
            "total_samples": self.metadata["total_samples"],
            "model_trained": self.metadata["model_trained"],
            "speakers": self.get_speaker_list()
        }
        
        if "model_info" in self.metadata:
            stats["model_info"] = self.metadata["model_info"]
        
        return stats

    def delete_speaker(self, speaker_name):
        """Kişiyi ve tüm verilerini sil"""
        import shutil
        speaker_id = speaker_name.lower().replace(" ", "_").replace("ç", "c").replace("ğ", "g").replace("ı", "i").replace("ö", "o").replace("ş", "s").replace("ü", "u")
        # Kişi metadata'da var mı?
        if speaker_id not in self.metadata["speakers"]:
            print(f"❌ Silinecek kişi bulunamadı: {speaker_name}")
            return False
        # Kişi klasörünü sil
        speaker_dir = os.path.join(self.speakers_dir, speaker_id)
        if os.path.exists(speaker_dir):
            try:
                shutil.rmtree(speaker_dir)
                print(f"🗑️ Kişi klasörü silindi: {speaker_dir}")
            except Exception as e:
                print(f"⚠️ Kişi klasörü silinemedi: {e}")
        # Metadata'dan sil
        sample_count = self.metadata["speakers"][speaker_id]["sample_count"]
        self.metadata["total_samples"] -= sample_count
        del self.metadata["speakers"][speaker_id]
        self.metadata["model_trained"] = False  # Model yeniden eğitilmeli
        self.save_metadata()
        print(f"✅ Kişi ve verileri silindi: {speaker_name}")
        return True

# Test fonksiyonu
def test_speaker_system():
    """Speaker identification sistemini test et"""
    print("🎤 Kişi Ses Tanıma Sistemi Testi")
    print("=" * 50)
    
    # Sistem oluştur
    speaker_system = SpeakerIdentificationSystem()
    
    # Test dosyalarını kontrol et
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        print("❌ recordings klasörü bulunamadı!")
        return
    
    # Belirli dosyaları ara
    menvoice_file = os.path.join(recordings_dir, "menvoice.wav")
    womenvoice_file = os.path.join(recordings_dir, "womenvoice.wav")
    
    if not os.path.exists(menvoice_file):
        print("❌ menvoice.wav dosyası bulunamadı!")
        return
    
    if not os.path.exists(womenvoice_file):
        print("❌ womenvoice.wav dosyası bulunamadı!")
        return
    
    print("✅ Test dosyaları bulundu:")
    print(f"   • menvoice.wav → Mert")
    print(f"   • womenvoice.wav → Ayşe")
    
    # 1. Mert'in sesini ekle - çoklu segment ile
    print(f"\n📝 1. ADIM: Mert'in ses örneğini ekliyorum (çoklu segment)...")
    success1 = speaker_system.add_multiple_samples_from_file("Mert", menvoice_file)
    
    # 2. Ayşe'nin sesini ekle - çoklu segment ile  
    print(f"\n📝 2. ADIM: Ayşe'nin ses örneğini ekliyorum (çoklu segment)...")
    success2 = speaker_system.add_multiple_samples_from_file("Ayşe", womenvoice_file)
    
    if not (success1 and success2):
        print("❌ Ses örnekleri eklenemedi!")
        return
    
    # 3. Model eğitimi
    print(f"\n📝 3. ADIM: Model eğitimi başlıyor...")
    if not speaker_system.train_model():
        print("❌ Model eğitimi başarısız!")
        return
    
    # 4. Test aşaması
    print(f"\n📝 4. ADIM: Tanıma testleri yapılıyor...")
    print("-" * 40)
    
    # Mert'in sesini test et
    print("🧪 Test 1: menvoice.wav (Beklenen: Mert)")
    speaker_name, confidence = speaker_system.identify_speaker(menvoice_file)
    result1 = "✅ DOĞRU" if speaker_name == "Mert" else "❌ YANLIŞ"
    print(f"   Sonuç: {speaker_name} (Güven: {confidence:.3f}) {result1}")
    
    # Ayşe'nin sesini test et
    print("\n🧪 Test 2: womenvoice.wav (Beklenen: Ayşe)")
    speaker_name, confidence = speaker_system.identify_speaker(womenvoice_file)
    result2 = "✅ DOĞRU" if speaker_name == "Ayşe" else "❌ YANLIŞ"
    print(f"   Sonuç: {speaker_name} (Güven: {confidence:.3f}) {result2}")
    
    # 5. İstatistikler
    print(f"\n📊 Sistem İstatistikleri:")
    stats = speaker_system.get_statistics()
    print(f"   • Toplam kişi: {stats['total_speakers']}")
    print(f"   • Toplam örnek: {stats['total_samples']}")
    print(f"   • Model eğitildi: {stats['model_trained']}")
    
    if "model_info" in stats:
        model_info = stats["model_info"]
        print(f"   • Model tipi: {model_info.get('model_type', 'Bilinmiyor')}")
        print(f"   • Model doğruluk: {model_info.get('accuracy', 0):.3f}")
    
    print(f"\n📂 Kayıtlı kişiler:")
    for speaker in stats['speakers']:
        print(f"   • {speaker['name']} ({speaker['sample_count']} örnek)")
    
    # 6. Ek test önerileri
    print(f"\n💡 Sistem geliştirme önerileri:")
    print(f"   • Her kişi için 3-5 farklı ses örneği ekleyin")
    print(f"   • Farklı cümleleri kaydedin")
    print(f"   • Farklı zamanlarda kayıt yapın")
    print(f"   • Model yeniden eğitilerek doğruluk artırılabilir")

def quick_test_mert_ayse():
    """Mert ve Ayşe için hızlı test"""
    print("🚀 Hızlı Test: Mert vs Ayşe")
    print("=" * 30)
    
    system = SpeakerIdentificationSystem()
    
    # Dosya yolları
    mert_file = "recordings/menvoice.wav"
    ayse_file = "recordings/womenvoice.wav"
    
    # Kontrol
    if not (os.path.exists(mert_file) and os.path.exists(ayse_file)):
        print("❌ Test dosyaları bulunamadı!")
        return
    
    # Ses örnekleri ekle
    system.add_voice_sample("Mert", mert_file)
    system.add_voice_sample("Ayşe", ayse_file)
    
    # Eğit
    if system.train_model():
        # Test
        m_name, m_conf = system.identify_speaker(mert_file)
        a_name, a_conf = system.identify_speaker(ayse_file)
        
        print(f"Mert testi: {m_name} ({m_conf:.3f})")
        print(f"Ayşe testi: {a_name} ({a_conf:.3f})")

if __name__ == "__main__":
    test_speaker_system()