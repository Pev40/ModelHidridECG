#!/bin/bash
# Script para matar procesos Python que están usando la GPU

echo "Procesos Python usando GPU:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "Matando procesos..."

# Obtener PIDs de procesos Python que usan GPU
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

for PID in $PIDS; do
    if [ ! -z "$PID" ] && [ "$PID" != "pid" ]; then
        echo "Matando proceso PID: $PID"
        kill -9 $PID 2>/dev/null || sudo kill -9 $PID 2>/dev/null
    fi
done

sleep 2

echo ""
echo "Verificando estado de GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Procesos restantes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

