#!/bin/bash
# Script di installazione ottimizzato per Mac M1
# Installa tutte le dipendenze necessarie per la versione ottimizzata

set -e  # Exit on error

echo "🚀 VIRAL REELS - INSTALLAZIONE OTTIMIZZATA PER MAC M1"
echo "=================================================="

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per logging colorato
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verifica sistema
echo
log_info "Verificando sistema..."

# Verifica macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_success "Sistema macOS rilevato"
    
    # Verifica architettura M1
    if [[ $(uname -m) == "arm64" ]]; then
        log_success "Architettura Apple Silicon (M1/M2) rilevata"
    else
        log_warning "Architettura Intel rilevata. Le ottimizzazioni sono specifiche per M1/M2"
    fi
else
    log_warning "Sistema non-macOS rilevato. Script ottimizzato per macOS M1"
fi

# Verifica Python
echo
log_info "Verificando Python..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_success "Python $PYTHON_VERSION trovato"
    
    # Verifica versione minima (3.8+)
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_success "Versione Python compatibile (>=3.8)"
    else
        log_error "Python 3.8+ richiesto. Versione attuale: $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python3 non trovato. Installare Python 3.8+ prima di continuare"
    exit 1
fi

# Verifica pip
if command -v pip3 &> /dev/null; then
    log_success "pip3 trovato"
else
    log_error "pip3 non trovato. Installare pip prima di continuare"
    exit 1
fi

# Verifica/Installa Homebrew
echo
log_info "Verificando Homebrew..."

if command -v brew &> /dev/null; then
    log_success "Homebrew trovato"
else
    log_warning "Homebrew non trovato. Installazione..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Aggiungi Homebrew al PATH per M1
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    log_success "Homebrew installato"
fi

# Installa FFmpeg
echo
log_info "Installando FFmpeg..."

if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    log_success "FFmpeg $FFMPEG_VERSION già installato"
else
    log_info "Installando FFmpeg via Homebrew..."
    brew install ffmpeg
    log_success "FFmpeg installato"
fi

# Verifica FFprobe
if command -v ffprobe &> /dev/null; then
    log_success "FFprobe disponibile"
else
    log_error "FFprobe non trovato. Reinstallare FFmpeg"
    exit 1
fi

# Crea ambiente virtuale
echo
log_info "Configurando ambiente virtuale Python..."

VENV_DIR="venv_optimized"

if [ -d "$VENV_DIR" ]; then
    log_warning "Ambiente virtuale esistente trovato. Rimuovendo..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
log_success "Ambiente virtuale creato e attivato"

# Aggiorna pip
log_info "Aggiornando pip..."
pip install --upgrade pip
log_success "pip aggiornato"

# Installa dipendenze ottimizzate per M1
echo
log_info "Installando dipendenze Python ottimizzate per M1..."

# PyTorch ottimizzato per M1
log_info "Installando PyTorch ottimizzato per Apple Silicon..."
pip install torch torchvision torchaudio

# Dipendenze core
log_info "Installando dipendenze core..."
pip install -r requirements_optimized.txt

# Dipendenze aggiuntive per test e monitoring
log_info "Installando dipendenze per test e monitoring..."
pip install matplotlib psutil

log_success "Tutte le dipendenze installate"

# Test installazione
echo
log_info "Testando installazione..."

# Test import principali
python3 -c "
import sys
modules = ['yt_dlp', 'faster_whisper', 'transformers', 'torch', 'numpy', 'soundfile', 'psutil']
failed = []

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
        failed.append(module)

if failed:
    print(f'\\n⚠️ Moduli falliti: {failed}')
    sys.exit(1)
else:
    print('\\n🎉 Tutti i moduli importati correttamente!')
"

if [ $? -eq 0 ]; then
    log_success "Test importazione completato"
else
    log_error "Test importazione fallito"
    exit 1
fi

# Test FFmpeg
log_info "Testando FFmpeg..."
ffmpeg -version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    log_success "FFmpeg funzionante"
else
    log_error "FFmpeg non funzionante"
    exit 1
fi

# Crea script di attivazione
echo
log_info "Creando script di attivazione..."

cat > activate_optimized.sh << 'EOF'
#!/bin/bash
# Script per attivare l'ambiente ottimizzato

echo "🚀 Attivando ambiente Viral Reels ottimizzato..."

# Attiva ambiente virtuale
source venv_optimized/bin/activate

# Verifica attivazione
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Ambiente virtuale attivato: $VIRTUAL_ENV"
else
    echo "❌ Errore attivazione ambiente virtuale"
    exit 1
fi

# Mostra info sistema
echo
echo "📊 INFO SISTEMA:"
echo "Python: $(python --version)"
echo "Memoria: $(python -c "import psutil; print(f'{psutil.virtual_memory().total/1024**3:.1f}GB')")"
echo "CPU: $(python -c "import psutil; print(f'{psutil.cpu_count()} cores')")"

echo
echo "🎬 Per avviare l'applicazione ottimizzata:"
echo "python viral_reels_optimized.py"
echo
echo "🧪 Per eseguire i test di performance:"
echo "python test_optimizations.py"
EOF

chmod +x activate_optimized.sh
log_success "Script di attivazione creato: activate_optimized.sh"

# Messaggio finale
echo
echo "=================================================="
log_success "INSTALLAZIONE COMPLETATA!"
echo "=================================================="
echo
echo "📋 PROSSIMI PASSI:"
echo
echo "1. Attiva l'ambiente ottimizzato:"
echo "   ${BLUE}source activate_optimized.sh${NC}"
echo
echo "2. Esegui i test di performance:"
echo "   ${BLUE}python test_optimizations.py${NC}"
echo
echo "3. Avvia l'applicazione ottimizzata:"
echo "   ${BLUE}python viral_reels_optimized.py${NC}"
echo
echo "📁 FILE CREATI:"
echo "   • viral_reels_optimized.py - Versione ottimizzata"
echo "   • requirements_optimized.txt - Dipendenze"
echo "   • test_optimizations.py - Test performance"
echo "   • OTTIMIZZAZIONI.md - Documentazione"
echo "   • venv_optimized/ - Ambiente virtuale"
echo "   • activate_optimized.sh - Script attivazione"
echo
echo "💡 SUGGERIMENTI:"
echo "   • Chiudi altre app durante il processing"
echo "   • Usa video <20 minuti per risultati ottimali"
echo "   • Monitora la memoria nell'interfaccia"
echo
log_success "Buon lavoro con la versione ottimizzata! 🎉"
