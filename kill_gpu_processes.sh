#!/bin/bash
# Script para matar procesos Python que están usando la GPU

echo "=== Procesos Python usando GPU antes de matar ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "=== Matando procesos... ==="

# Matar procesos específicos si existen
if kill -9 2125 2>/dev/null || sudo kill -9 2125 2>/dev/null; then
    echo "Proceso 2125 terminado"
else
    echo "Proceso 2125 no encontrado o ya terminado"
fi

if kill -9 2592 2>/dev/null || sudo kill -9 2592 2>/dev/null; then
    echo "Proceso 2592 terminado"
else
    echo "Proceso 2592 no encontrado o ya terminado"
fi

# Esperar un poco para que se libere la memoria
sleep 3

echo ""
echo "=== Estado de GPU después de matar procesos ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv

echo ""
echo "=== Procesos restantes usando GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
