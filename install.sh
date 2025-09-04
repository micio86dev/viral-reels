#!/bin/bash
# Optimized installation script for Mac M1
# Installs all necessary dependencies for the optimized version

set -e  # Exit on error

echo "🚀 VIRAL REELS - OPTIMIZED INSTALLATION FOR MAC M1"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for colored logging
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

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is designed for macOS only"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    log_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add to PATH for Apple Silicon Macs
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

log_success "Homebrew is ready"

# Install system dependencies
log_info "Installing system dependencies..."
brew install python@3.13 ffmpeg

# Create virtual environment
log_info "Creating virtual environment..."
python3.13 -m venv venv
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
log_info "Installing Python dependencies..."
pip install -r requirements.txt

# Create desktop launcher
log_info "Creating desktop launcher..."
SCRIPT_DIR=$(pwd)
cat > ViralReels.command << EOF
#!/bin/bash
cd "$SCRIPT_DIR"
source venv/bin/activate
python viral_reels.py
deactivate
EOF

chmod +x ViralReels.command

# Move launcher to Desktop
if [ -d "$HOME/Desktop" ]; then
    cp ViralReels.command "$HOME/Desktop/"
    log_success "Launcher copied to Desktop: ViralReels.command"
fi

log_success "Installation complete!"
echo ""
echo "🚀 To run the application:"
echo "   1. Double-click ViralReels.command on your Desktop"
echo "   2. Or run: ./ViralReels.command"
echo ""
echo "📁 Output files will be saved to: ~/Desktop/ViralReels/"
echo ""
echo "💡 For best performance on Mac M1:"
echo "   - Close other applications during processing"
echo "   - Ensure you have at least 4GB free RAM"
echo "   - The first run may take longer as models are downloaded"