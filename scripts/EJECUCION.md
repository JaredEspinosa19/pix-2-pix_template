# Cómo Ejecutar los Scripts

Los scripts en esta carpeta están diseñados para ejecutarse desde **cualquier ubicación** y automáticamente cambiarán al directorio raíz del proyecto.

## Formas de Ejecutar

### Opción 1: Desde la raíz del proyecto (Recomendado)

```bash
# Método directo
./scripts/train_model.sh canny

# Usando el wrapper
./run_script.sh train_model canny
```

### Opción 2: Desde dentro de la carpeta scripts/

```bash
cd scripts/
./train_model.sh canny
```

### Opción 3: Desde cualquier ubicación

```bash
bash /ruta/completa/al/proyecto/scripts/train_model.sh canny
```

## Cómo Funcionan

Todos los scripts incluyen esta línea al inicio:

```bash
cd "$(dirname "$0")/.." || exit 1
```

Esto significa que:
1. `$(dirname "$0")` obtiene la carpeta donde está el script (scripts/)
2. `/..' sube un nivel al directorio raíz del proyecto
3. `cd` cambia a ese directorio
4. El resto del script se ejecuta desde la raíz

Por lo tanto, **todas las rutas relativas funcionarán correctamente** sin importar desde dónde ejecutes el script.

## Ejemplos

**Entrenar desde raíz:**
```bash
./scripts/train_model.sh canny
```

**Entrenar desde scripts/:**
```bash
cd scripts
./train_model.sh canny
```

**Entrenar desde otra carpeta:**
```bash
cd dataset
../scripts/train_model.sh canny
```

Todos funcionan igual porque el script automáticamente se posiciona en la raíz.

## Rutas en los Scripts

Los scripts usan rutas relativas desde la raíz del proyecto:

```bash
INPUT_PATH="dataset/${ALGORITMO}"          # Correcto
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"  # Correcto
python train.py ...                        # Correcto (train.py está en raíz)
```

**No necesitas modificar estas rutas** - funcionan automáticamente gracias al `cd` inicial.
