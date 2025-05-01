import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_NAME = 'anpr_uploads'
UPLOAD_FOLDER_PATH = os.path.join(APP_ROOT, UPLOAD_FOLDER_NAME)

PLATE_MODEL_PATH = os.path.join(APP_ROOT, 'models/pd_traific_v2_mix.pt')
CHAR_SEG_MODEL_PATH = os.path.join(APP_ROOT, 'models/sg_traific_v12.pt')
CHAR_REC_MODEL_PATH = os.path.join(APP_ROOT, 'models/char_traific_v3.pth')

FONT_PATH = "F:/development/python/Noto_Sans_Devanagari/NotoSansDevanagari-Regular.ttf"

CLASS_LABELS = [
    'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा',
    'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '0', '१', '२', '३', '४', '५',
    '६', '७', '८', '९'
]
NUM_CLASSES = len(CLASS_LABELS)

PLATE_DETECT_CONF = 0.4
CHAR_SEG_CONF = 0.3
CHAR_REC_CONF_THRESHOLD = 0.4 

VIDEO_FRAME_SKIP = 0

DESKEW_MIN_PLATE_HEIGHT = 15
DESKEW_MIN_PLATE_WIDTH = 30

CHAR_ORDERING_HEIGHT_FRACTION = 0.6
CHAR_ORDERING_LINE_GAP_FACTOR = 0.5

FLASK_SECRET_KEY = 'your_very_secret_key_change_me'
MAX_CONTENT_LENGTH = 32 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv'}