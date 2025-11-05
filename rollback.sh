#!/bin/bash

echo "🔄 INICIANDO ROLLBACK..."

# Encontrar backup más reciente
LATEST_BACKUP=$(ls -td backups/*/ | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No se encontró backup"
    exit 1
fi

echo "📦 Usando backup: $LATEST_BACKUP"

# Restaurar archivo
cp "$LATEST_BACKUP/analizador_optimizado.py" backend/analizador_optimizado.py

echo "✅ ROLLBACK COMPLETADO"
echo "📝 Backend restaurado a versión anterior"
echo ""
echo "Si estabas ejecutando Streamlit, reinicia con:"
echo "  streamlit run frontend/streamlit_app.py"

