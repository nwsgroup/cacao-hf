#!/bin/bash

# Configuración de entrenamiento
set -e

# Parámetros de entrenamiento uniformes
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=1e-4
COOLDOWN_TIME=30  # segundos entre entrenamientos

# Crear directorio de logs
mkdir -p logs

# Función para logging con timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Función para mostrar tiempo transcurrido
show_elapsed_time() {
    local start_time=$1
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Verificar dependencias
if ! command -v jq &> /dev/null; then
    echo "Error: jq no está instalado. Instálalo con: sudo apt install jq"
    exit 1
fi

if [ ! -f "best.json" ]; then
    echo "Error: archivo best.json no encontrado"
    exit 1
fi

if [ ! -f "train.sh" ]; then
    echo "Error: archivo train.sh no encontrado"
    exit 1
fi

# Obtener lista de modelos
MODELS=( $(jq -r '.models | keys[]' best.json) )

# Mostrar configuración
echo "================================================================"
echo "           ENTRENAMIENTO MASIVO DE MODELOS DE CACAO"
echo "================================================================"
echo "Modelos encontrados: ${#MODELS[@]}"
echo "Configuración de entrenamiento:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Épocas: $EPOCHS"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Tiempo de enfriamiento: ${COOLDOWN_TIME}s"
echo ""
echo "Lista de modelos:"
printf ' - %s\n' "${MODELS[@]}"
echo ""

# Confirmación del usuario
read -p "¿Continuar con el entrenamiento de ${#MODELS[@]} modelos? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Entrenamiento cancelado."
    exit 0
fi

# Variables para estadísticas
TOTAL_START_TIME=$(date +%s)
SUCCESSFUL_MODELS=()
FAILED_MODELS=()
MODEL_TIMES=()

echo ""
echo "================================================================"
echo "Iniciando entrenamiento secuencial..."
echo "================================================================"

# Loop principal de entrenamiento
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUMBER=$((i + 1))
    
    echo ""
    log_message "[$MODEL_NUMBER/${#MODELS[@]}] Iniciando entrenamiento: $MODEL"
    echo "================================================================"
    
    # Tiempo de inicio del modelo
    MODEL_START_TIME=$(date +%s)
    
    # Obtener información del modelo del JSON
    MODEL_INFO=$(jq -r ".models.$MODEL.description" best.json)
    MODEL_NAME=$(jq -r ".models.$MODEL.name" best.json)
    
    echo "Modelo: $MODEL_NAME"
    echo "Descripción: $MODEL_INFO"
    echo "Parámetros: batch_size=$BATCH_SIZE, epochs=$EPOCHS, lr=$LEARNING_RATE"
    echo ""
    
    # Ejecutar entrenamiento
    if bash train.sh -m "$MODEL" -b "$BATCH_SIZE" -e "$EPOCHS" -l "$LEARNING_RATE" 2>&1 | tee "logs/training_${MODEL}.log"; then
        MODEL_END_TIME=$(date +%s)
        MODEL_ELAPSED=$(show_elapsed_time $MODEL_START_TIME)
        MODEL_TIMES+=("$MODEL: $MODEL_ELAPSED")
        SUCCESSFUL_MODELS+=("$MODEL")
        
        log_message "✅ [$MODEL_NUMBER/${#MODELS[@]}] Completado: $MODEL (Tiempo: $MODEL_ELAPSED)"
        
        # Verificar que el modelo se guardó
        OUTPUT_DIR="./outputs/outputs_${MODEL}"
        if [ -f "$OUTPUT_DIR/model.safetensors" ]; then
            MODEL_SIZE=$(du -h "$OUTPUT_DIR/model.safetensors" | cut -f1)
            log_message "   Modelo guardado: $OUTPUT_DIR/model.safetensors ($MODEL_SIZE)"
        fi
        
    else
        MODEL_END_TIME=$(date +%s)
        MODEL_ELAPSED=$(show_elapsed_time $MODEL_START_TIME)
        FAILED_MODELS+=("$MODEL")
        
        log_message "❌ [$MODEL_NUMBER/${#MODELS[@]}] Falló: $MODEL (Tiempo: $MODEL_ELAPSED)"
        log_message "   Ver detalles en: logs/training_${MODEL}.log"
    fi
    
    # Tiempo de enfriamiento entre modelos (excepto el último)
    if [ $MODEL_NUMBER -lt ${#MODELS[@]} ]; then
        log_message "Enfriando GPU por ${COOLDOWN_TIME} segundos..."
        
        # Mostrar countdown
        for ((j=COOLDOWN_TIME; j>0; j--)); do
            printf "\rTiempo restante: %02d segundos" $j
            sleep 1
        done
        printf "\r✅ Enfriamiento completado          \n"
    fi
done

# Estadísticas finales
TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$(show_elapsed_time $TOTAL_START_TIME)

echo ""
echo "================================================================"
echo "                    RESUMEN FINAL"
echo "================================================================"
echo "Tiempo total de entrenamiento: $TOTAL_ELAPSED"
echo "Modelos exitosos: ${#SUCCESSFUL_MODELS[@]}/${#MODELS[@]}"
echo "Modelos fallidos: ${#FAILED_MODELS[@]}/${#MODELS[@]}"
echo ""

# Mostrar modelos exitosos
if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo "✅ Modelos entrenados exitosamente:"
    for i in "${!SUCCESSFUL_MODELS[@]}"; do
        echo "   $((i+1)). ${SUCCESSFUL_MODELS[$i]}"
    done
    echo ""
fi

# Mostrar modelos fallidos
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "❌ Modelos que fallaron:"
    for i in "${!FAILED_MODELS[@]}"; do
        echo "   $((i+1)). ${FAILED_MODELS[$i]}"
    done
    echo ""
fi

# Mostrar tiempos por modelo
echo "⏱️  Tiempos de entrenamiento por modelo:"
for time_entry in "${MODEL_TIMES[@]}"; do
    echo "   $time_entry"
done

echo ""
echo "📁 Logs disponibles en: ./logs/"
echo "📁 Modelos guardados en: ./outputs/"

# Crear resumen en archivo
SUMMARY_FILE="logs/training_summary_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Resumen de entrenamiento masivo - $(date)"
    echo "========================================"
    echo "Configuración:"
    echo "- Batch Size: $BATCH_SIZE"
    echo "- Épocas: $EPOCHS"
    echo "- Learning Rate: $LEARNING_RATE"
    echo ""
    echo "Resultados:"
    echo "- Tiempo total: $TOTAL_ELAPSED"
    echo "- Exitosos: ${#SUCCESSFUL_MODELS[@]}/${#MODELS[@]}"
    echo "- Fallidos: ${#FAILED_MODELS[@]}/${#MODELS[@]}"
    echo ""
    echo "Modelos exitosos:"
    printf '%s\n' "${SUCCESSFUL_MODELS[@]}"
    echo ""
    echo "Modelos fallidos:"
    printf '%s\n' "${FAILED_MODELS[@]}"
    echo ""
    echo "Tiempos por modelo:"
    printf '%s\n' "${MODEL_TIMES[@]}"
} > "$SUMMARY_FILE"

log_message "📋 Resumen guardado en: $SUMMARY_FILE"

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "🎉 ¡Todos los modelos se entrenaron exitosamente!"
    exit 0
else
    echo ""
    echo "⚠️  Algunos modelos fallaron. Revisa los logs para más detalles."
    exit 1
fi