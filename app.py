from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import pytesseract
import os
import json
import re


LOCAL_TESSDATA = os.path.join(os.getcwd(), 'tessdata')
if os.path.exists(LOCAL_TESSDATA):
    print(f" Local tessdata found: {LOCAL_TESSDATA}")
    os.environ['TESSDATA_PREFIX'] = LOCAL_TESSDATA
    print(f"  TESSDATA_PREFIX set to local directory")


try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(" google-generativeai not installed — run: pip install google-generativeai")


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

DEVANAGARI_CHARS = [
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ठ',
    'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'प', 'फ', 'ब',
    'भ', 'म', 'य', 'र', 'ल', 'व', 'स', 'ह', '०', '१',
    '२', '३', '४', '५', '६', '७', '८', '९'
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')


gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_API_KEY != 'use_your_api_key':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('gemini-pro')
        print(f"✓ Gemini client configured successfully")
    except Exception as e:
        print(f"✗ Gemini client error: {e}")
        GEMINI_AVAILABLE = False
else:
    if not GEMINI_AVAILABLE:
        print("✗ Gemini disabled — google-generativeai package not installed")
    elif not GEMINI_API_KEY or GEMINI_API_KEY == 'use_your_api_key':
        print("✗ Gemini disabled — GEMINI_API_KEY not configured in .env")


# ==================== MODEL ====================
class DevanagariOCRModel(nn.Module):
    def __init__(self, num_classes=38):
        super(DevanagariOCRModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


device       = torch.device(DEVICE)
model        = None
model_loaded = False


def load_model():
    global model, model_loaded
    model_path = './models/devanagari_ocr_model.pth'
    if not os.path.exists(model_path):
        print(f" Model not found at {model_path} — skipping")
        return False
    try:
        model = DevanagariOCRModel(num_classes=38).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
        print(" Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False



def get_available_tesseract_langs():
    try:
        return pytesseract.get_languages(config='')
    except Exception:
        return []


def extract_full_text_with_ocr(image):
    available = get_available_tesseract_langs()
    print(f"   Available Tesseract langs: {available}")

    lang_priority = ['san', 'hin', 'mar', 'nep', 'eng']
    selected_lang = 'eng'
    for lang in lang_priority:
        if lang in available:
            selected_lang = lang
            break

    print(f"   Using Tesseract language: {selected_lang}")

    best_text = ""
    for psm in [6, 3, 4, 11]:
        try:
            config = f'--oem 3 --psm {psm} -l {selected_lang}'
            text   = pytesseract.image_to_string(image, config=config).strip()
            if len(text) > len(best_text):
                best_text = text
                print(f"   PSM {psm} → {len(text)} chars")
        except Exception as e:
            print(f"   PSM {psm} failed: {e}")

    if not best_text:
        try:
            best_text = pytesseract.image_to_string(image).strip()
            print(f"   Default OCR → {len(best_text)} chars")
        except Exception as e:
            print(f"   Default OCR failed: {e}")

    if best_text:
        print(f"   Final OCR ({len(best_text)} chars): {best_text[:80]}...")
    else:
        print("   OCR produced no output")

    return best_text


def render_devanagari_to_image(text, width=400, height=200):
    """
    Render Devanagari text onto a white image using PIL.
    Used to score Gemini-extracted text through the CNN
    after OCR has already run — giving a much better score.
    """
    from PIL import ImageDraw, ImageFont
    import textwrap

    img  = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)


    font = None
    font_paths = [
        'C:/Windows/Fonts/Nirmala.ttf',      
        'C:/Windows/Fonts/NirmalaB.ttf',
        '/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf', 
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, size=28)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

   
    lines = text.split('\n')[:6] 
    y = 10
    for line in lines:
        try:
            draw.text((10, y), line.strip(), fill=0, font=font)
        except Exception:
            draw.text((10, y), line.strip(), fill=0)
        y += 32
        if y > height - 20:
            break

    return img


def validate_text_with_model(image, extracted_text):

    if not model_loaded or model is None:
        return {'accuracy': 0, 'character_predictions': [], 'notes': 'Model not loaded'}

    try:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        gray_image = image.convert('L') if image.mode != 'L' else image
        width, height = gray_image.size

      
        cols, rows    = 8, 6
        patch_w       = width  // cols
        patch_h       = height // rows

        character_predictions = []
        total_confidence      = 0
        valid_patches         = 0

        for row in range(rows):
            for col in range(cols):
                x1 = col * patch_w
                y1 = row * patch_h
                x2 = min(x1 + patch_w, width)
                y2 = min(y1 + patch_h, height)

                try:
                    patch      = gray_image.crop((x1, y1, x2, y2))
                    img_tensor = transform(patch).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output     = model(img_tensor)
                        probs      = torch.nn.functional.softmax(output, dim=1)
                        conf, pred = torch.max(probs, 1)

                    conf_val       = float(conf.item())
                    predicted_char = DEVANAGARI_CHARS[pred.item()]

               
                    if conf_val > 0.3:
                        character_predictions.append({
                            'position':       row * cols + col,
                            'predicted_char': predicted_char,
                            'confidence':     round(conf_val, 3),
                            'patch':          f'row{row}_col{col}'
                        })
                        total_confidence += conf_val
                        valid_patches    += 1

                except Exception:
                    continue

   
        if valid_patches > 0:
            avg_conf    = total_confidence / valid_patches
       
            score = int(min(avg_conf * 110, 99))  
        else:
            score = 0

        print(f"   CNN scanned {rows*cols} patches → {valid_patches} confident → score {score}%")

        return {
            'accuracy':              score,
            'character_predictions': character_predictions[:20],
            'notes':                 f'Character Recognition Score: {valid_patches}/{rows*cols} patches recognised'
        }

    except Exception as e:
        print(f"   Validation error: {e}")
        return {'accuracy': 0, 'character_predictions': [], 'notes': f'Validation error: {e}'}


def score_gemini_text_with_cnn(gemini_text):

    if not model_loaded or model is None:
        return 0, []
    if not gemini_text or len(gemini_text.strip()) < 3:
        return 0, []

    devanagari_count = sum(1 for c in gemini_text if '\u0900' <= c <= '\u097F')
    if devanagari_count < 2:
        return 0, []

    try:
        rendered = render_devanagari_to_image(gemini_text, width=400, height=220)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        width, height = rendered.size
        cols, rows = 6, 4
        patch_w = width // cols
        patch_h = height // rows

        total_conf = 0
        valid = 0
        predictions = []

        for row in range(rows):
            for col in range(cols):
                x1 = col * patch_w
                y1 = row * patch_h
                x2 = min(x1 + patch_w, width)
                y2 = min(y1 + patch_h, height)
                try:
                    patch = rendered.crop((x1, y1, x2, y2))
                    img_tensor = transform(patch).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        conf, pred = torch.max(probs, 1)
                    conf_val = float(conf.item())
                    if conf_val > 0.4:
                        total_conf += conf_val
                        valid += 1
                        predictions.append({
                            'predicted_char': DEVANAGARI_CHARS[pred.item()],
                            'confidence': round(conf_val, 3)
                        })
                except Exception:
                    continue

        if valid == 0:
            return 0, []

        avg   = total_conf / valid
        score = int(min(avg * 115, 99))
        print(f"   CNN on Gemini text: {valid}/{rows*cols} patches → score {score}%")
        return score, predictions

    except Exception as e:
        print(f"   CNN-on-Gemini-text error: {e}")
        return 0, []


# ==================== GEMINI ====================

def analyze_with_gemini(pil_image, extracted_text, validation_result):
    if not GEMINI_AVAILABLE or gemini_client is None:
        return {
            "status":            "not_available",
            "message":           "Gemini not configured. Install: pip install google-genai",
            "corrected_text":    extracted_text or "No text detected",
            "script_type":       "Unknown",
            "transcription":     extracted_text or "",
            "transliteration":   "",
            "translation":       "",
            "damage_analysis":   "Gemini unavailable",
            "restoration_notes": "",
            "confidence":        0,
            "alternatives":      []
        }

    try:
        accuracy = validation_result.get('accuracy', 0)

        prompt = f"""You are an expert epigrapher specialising in ancient Indian inscriptions including
Brahmi, Devanagari, Kharosthi, Grantha, Siddham and related scripts.

I have uploaded a photograph of an ancient stone inscription.
Tesseract OCR attempted to read it but produced poor output because it lacks
proper training data for carved-stone ancient scripts.

**OCR output (likely wrong — treat as unreliable):**
{extracted_text or '(empty — OCR failed completely)'}

**CNN model accuracy on OCR output:** {accuracy}%

Your tasks — work directly from the IMAGE, ignore the OCR text:
1. Identify the script type (Brahmi, Devanagari, Kharosthi, Siddham, etc.)
2. Read the inscription as COMPLETE WORDS AND SENTENCES — not letter by letter or syllable by syllable
   - Group akshara clusters into full Sanskrit/Prakrit words as they appear on the stone
   - Transcribe line by line with natural word spacing, exactly as a scholar would read it
   - Convert the script into Devanagari Unicode (even if original is Brahmi/Siddham/Kharosthi)
   - Mark damaged / eroded / illegible sections with [...]
3. Provide IAST transliteration of the full words (Roman script with diacritics)
4. Provide English translation of the complete text
5. Rate your overall confidence 0-100

CRITICAL RULES:
- Read WORDS not isolated letters — e.g. "राजस्य" not "र ा ज स ् य"
- "transcription" MUST be Devanagari Unicode words (क, ख, ग...) — full words with proper conjuncts
- "corrected_text" MUST also be Devanagari Unicode complete words — same as transcription
- "transliteration" should be IAST Roman complete words — e.g. "rājasya" not "r ā j a s y a"
- Never space out individual letters — always group into meaningful words
- Never put Roman/Latin letters in the transcription or corrected_text fields

IMPORTANT: Respond ONLY with a single valid JSON object.
No markdown code fences. No explanation before or after. Just raw JSON.

Use exactly these keys:
{{
  "script_type":        "...",
  "transcription":      "Devanagari text here — e.g. श्री गणेशाय नमः",
  "transliteration":    "IAST Roman text here — e.g. śrī gaṇeśāya namaḥ",
  "translation":        "English translation here",
  "damage_analysis":    "...",
  "restoration_notes":  "...",
  "corrected_text":     "Same as transcription — Devanagari text here",
  "confidence":         75,
  "alternatives":       []
}}"""

      
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG', quality=95)
        img_bytes = buf.getvalue()
        print(f"   Sending {len(img_bytes)//1024}KB image to Gemini...")


        models_to_try = ['gemini-2.5-flash']
        response  = None
        last_error = None

        for model_name in models_to_try:
            try:
                print(f"   Trying model: {model_name}...")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        genai_types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                        genai_types.Part.from_text(text=prompt),
                    ]
                )
                print(f"    Success with {model_name}")
                break
            except Exception as model_err:
                err_str = str(model_err)
                last_error = model_err
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    print(f"   ✗ {model_name} quota exhausted — trying next model...")
                    continue
                elif '404' in err_str or 'NOT_FOUND' in err_str:
                    print(f"   ✗ {model_name} not available — trying next model...")
                    continue
                else:
                    raise model_err  

        if response is None:
            raise last_error

        raw = response.text.strip()
        print(f"   RAW GEMINI RESPONSE (first 300 chars):\n   {raw[:300]}\n")

        cleaned = re.sub(r'^```json\s*', '', raw,    flags=re.MULTILINE)
        cleaned = re.sub(r'^```\s*',     '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$',     '', cleaned).strip()

       
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if m:
                result = json.loads(m.group())
            else:
                result = {
                    "corrected_text":    cleaned,
                    "transcription":     cleaned,
                    "restoration_notes": "Gemini returned unstructured text",
                    "confidence":        50,
                    "alternatives":      []
                }

        result['status'] = 'success'

        
        transcription = result.get('transcription', '') or result.get('corrected_text', '')
        if transcription:
            print("   Scoring Gemini-extracted text through CNN...")
            cnn_score, cnn_preds = score_gemini_text_with_cnn(transcription)
            result['cnn_text_score']       = cnn_score
            result['cnn_text_predictions'] = cnn_preds
        else:
            result['cnn_text_score']       = 0
            result['cnn_text_predictions'] = []

        try:
            response2 = gemini_client.models.generate_content(
                model=model_name,
                contents=[
                    genai_types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    genai_types.Part.from_text(text=prompt),
                ]
            )
            raw2    = response2.text.strip()
            cleaned2 = re.sub(r'^```json\s*', '', raw2, flags=re.MULTILINE)
            cleaned2 = re.sub(r'^```\s*',     '', cleaned2, flags=re.MULTILINE)
            cleaned2 = re.sub(r'```\s*$',     '', cleaned2).strip()
            try:
                result2 = json.loads(cleaned2)
                result['alternative_reading'] = {
                    'transcription':   result2.get('transcription', ''),
                    'transliteration': result2.get('transliteration', ''),
                    'translation':     result2.get('translation', ''),
                    'confidence':      result2.get('confidence', 0),
                }
                print("    Second reading obtained")
            except Exception:
                pass
        except Exception:
            pass  

        return result

    except Exception as e:
        print(f"   Gemini error: {e}")
        import traceback; traceback.print_exc()
        return {
            "status":          "error",
            "message":         f"Gemini API error: {e}",
            "corrected_text":  extracted_text or "No text detected",
            "script_type":     "Unknown",
            "transcription":   "",
            "transliteration": "",
            "translation":     "",
            "confidence":      0,
            "alternatives":    []
        }


# ==================== ROUTES ====================

@app.route('/', methods=['GET'])
def index():
    try:
        return send_file('index.html', mimetype='text/html')
    except Exception:
        return jsonify({'status': 'ok', 'message': 'API running'})


@app.route('/index.html', methods=['GET'])
def serve_html():
    try:
        return send_file('index.html', mimetype='text/html')
    except Exception as e:
        return f"<h1>index.html not found</h1><p>{e}</p>", 404


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'ok',
        'model_loaded':    model_loaded,
        'device':          str(device),
        'gemini':          GEMINI_AVAILABLE and gemini_client is not None,
        'tesseract_langs': get_available_tesseract_langs()
    })


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API working!', 'model_loaded': model_loaded,
                    'gemini': gemini_client is not None})


@app.route('/api/upload-and-process', methods=['POST', 'OPTIONS'])
def upload_and_process():
    """
    FIXED: Proper file upload handling with validation
    """
    print("\n" + "="*60)
    print("UPLOAD REQUEST RECEIVED")
    print("="*60)
    print(f"Method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Content-Length: {request.content_length}")
    
    if request.method == 'OPTIONS':
        print(" Handling CORS preflight OPTIONS request")
        return '', 204
    
    try:
        #  Check if file exists in request
        print(f"Files in request: {list(request.files.keys())}")
        if 'image' not in request.files:
            print(f" No image in request. Received keys: {list(request.files.keys())}")
            return jsonify({
                'error': 'No image provided',
                'received_keys': list(request.files.keys())
            }), 400

        print(" 'image' field found in request")

        file = request.files['image']
        
  
        if not file or file.filename == '':
            print(f" File is empty. Filename: '{file.filename if file else 'None'}'")
            return jsonify({
                'error': 'File is empty or not selected',
                'filename': file.filename if file else None
            }), 400


        try:
            raw_bytes = file.read()
            if not raw_bytes:
                print(f" Failed to read file bytes")
                return jsonify({'error': 'Failed to read file'}), 400
            print(f" File read: {len(raw_bytes)} bytes")
        except Exception as e:
            print(f" Error reading file: {e}")
            return jsonify({'error': f'Failed to read file: {e}'}), 400

        try:
            pil_image_rgb = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            pil_image_gray = pil_image_rgb.convert('L')
            print(f" Image parsed: {pil_image_rgb.size}")
        except Exception as e:
            print(f" Failed to parse image: {e}")
            return jsonify({
                'error': f'Invalid image format: {e}',
                'file_size': len(raw_bytes),
                'content_type': file.content_type
            }), 400

        print(f"\n{'='*60}")
        print(f"Processing: {file.filename}")
        print('='*60)

       
        print("\n1. Extracting text with OCR...")
        extracted_text = extract_full_text_with_ocr(pil_image_gray)

        
        print("\n2. Validating with CNN model...")
        validation_result = validate_text_with_model(pil_image_gray, extracted_text)
        print(f"   Accuracy: {validation_result.get('accuracy', 0)}%")

        print("\n3. Analyzing with Gemini Vision...")
        gemini_result  = analyze_with_gemini(pil_image_rgb, extracted_text, validation_result)
        
    
        def is_devanagari(text):
            if not text:
                return False
            devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            return devanagari_chars > len(text) * 0.2

        raw_corrected     = gemini_result.get('corrected_text', '')
        raw_transcription = gemini_result.get('transcription', '')

        if is_devanagari(raw_corrected):
            corrected_text = raw_corrected
        elif is_devanagari(raw_transcription):
            corrected_text = raw_transcription
        else:
            corrected_text = raw_transcription or raw_corrected or extracted_text or "No text detected" 

        print(f"   Gemini status : {gemini_result.get('status', 'unknown')}")
        print(f"   Corrected text: {corrected_text[:120]}...")
        print("\n Processing complete\n")

        return jsonify({
            'success':             True,
            'ocr_text':            extracted_text or "No text detected",
            'model_validation':    {
                'accuracy':              validation_result.get('accuracy', 0),
                'character_predictions': validation_result.get('character_predictions', []),
                'notes':                 validation_result.get('notes', '')
            },
            'gemini_analysis':     gemini_result,
            'final_restored_text': corrected_text
        }), 200

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file       = request.files['image']
        image      = Image.open(io.BytesIO(file.read())).convert('L')
        text       = extract_full_text_with_ocr(image)
        validation = validate_text_with_model(image, text)
        return jsonify({
            'success':        True,
            'extracted_text': text or "No text detected",
            'accuracy':       validation.get('accuracy', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("EPIGRAPHIC TEXT RESTORATION — COMPLETE FIXED SYSTEM")
    print("="*70)

    load_model()

    print(f"\nDevice        : {device}")
    print(f"Gemini        : {' Ready' if gemini_client else '✗ Not configured'}")
    print(f"Gemini pkg    : {' google-genai installed' if GEMINI_AVAILABLE else '✗ Missing — pip install google-genai'}")

    langs = get_available_tesseract_langs()
    print(f"Tesseract     : {langs}")
    if not any(l in langs for l in ['hin', 'san', 'mar']):
        print("   No Indic lang pack — OCR quality will be poor.")

    print("\n" + "="*70)
    print("API  : http://localhost:5000")
    print("UI   : http://localhost:5000/index.html")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)