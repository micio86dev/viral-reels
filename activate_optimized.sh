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
