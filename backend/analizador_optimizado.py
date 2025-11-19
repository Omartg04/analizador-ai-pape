# analizador_politica_social.py Versión 2.0
"""
Sistema de Análisis para Política Social - VERSIÓN MODULAR MEJORADA
Con capacidades específicas de análisis de brechas y elegibilidad
"""

import email
import pandas as pd
import numpy as np
import json
import warnings
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from openai import OpenAI
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

def get_api_key():
    """
    Obtiene API key de forma segura desde variables de entorno
    
    Returns:
        str: API key válida
        
    Raises:
        EnvironmentError: Si la variable de entorno no está configurada
        ValueError: Si el formato de la API key es inválido
    """
    # Intentar obtener de variable de entorno
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # Si no existe, dar instrucciones claras
    if not api_key:
        raise EnvironmentError(
            "\n" + "="*60 + "\n"
            "❌ ERROR: DEEPSEEK_API_KEY no configurada\n"
            "="*60 + "\n"
            "Por favor, configura tu API key con uno de estos métodos:\n\n"
            "OPCIÓN 1 - En la terminal (temporal):\n"
            "  export DEEPSEEK_API_KEY='tu-key-aqui'\n\n"
            "OPCIÓN 2 - En tu .bashrc o .zshrc (permanente):\n"
            "  echo 'export DEEPSEEK_API_KEY=\"tu-key-aqui\"' >> ~/.bashrc\n"
            "  source ~/.bashrc\n\n"
            "OPCIÓN 3 - En un archivo .env:\n"
            "  1. Crea archivo .env en el directorio del proyecto\n"
            "  2. Agrega: DEEPSEEK_API_KEY=tu-key-aqui\n"
            "  3. Instala: pip install python-dotenv\n"
            "  4. Al inicio del script: from dotenv import load_dotenv; load_dotenv()\n"
            "="*60
        )
    
    # Validar formato básico de la API key
    api_key = api_key.strip()
    
    if not api_key.startswith('sk-'):
        raise ValueError(
            "❌ ERROR: API key inválida\n"
            "Las API keys de DeepSeek deben comenzar con 'sk-'\n"
            f"Tu key comienza con: '{api_key[:5]}...'"
        )
    
    if len(api_key) < 20:
        raise ValueError(
            "❌ ERROR: API key muy corta\n"
            "Las API keys válidas tienen al menos 20 caracteres"
        )
    
    print("✅ API key válida cargada correctamente")
    return api_key

# ============================================================================
# 1. CLASE DE INTEGRACIÓN DE DATOS (VERSIÓN LIMPIA Y COMPLETA)
# ============================================================================
class DataIntegrator:
    """Clase para integrar y preparar los datasets de análisis"""
    
    def __init__(self):
        self.df_completo = None
        self.df_personas = None
        self.df_hogares = None
        
    def cargar_y_unir_datasets(self, ruta_base: str = None):
        """
        Carga y une los datasets con validación de integridad
        VERSIÓN MEJORADA: Detecta automáticamente la ruta correcta
        """
        
        # Si no se proporciona ruta, detectarla automáticamente
        if ruta_base is None:
            ruta_base = self._detectar_ruta_datos()
            print(f"🔍 Ruta detectada automáticamente: {ruta_base}")
        else:
            print(f"🔍 Cargando datos desde: {ruta_base}")
        
        # Validar que la ruta existe
        if not os.path.exists(ruta_base):
            raise FileNotFoundError(
                f"\n❌ Carpeta de datos no encontrada: {ruta_base}\n"
                f"   Ruta absoluta: {os.path.abspath(ruta_base)}\n"
                f"   Verifica que existe la carpeta 01_data con los archivos CSV"
            )        
        
        df_hogares = pd.read_csv(f"{ruta_base}/CaracteristicasHogar.csv")
        df_caracteristicas = pd.read_csv(f"{ruta_base}/CaracteristicasPersona.csv")
        df_carencias = pd.read_csv(f"{ruta_base}/CarenciasPersona.csv") 
        df_intervenciones = pd.read_csv(f"{ruta_base}/IntervencionesPotencialesPAPEPersona.csv")
        
        print("📊 Dimensiones de datasets originales:")
        print(f"  Hogares: {df_hogares.shape}")
        print(f"  Características Persona: {df_caracteristicas.shape}")
        print(f"  Carencias Persona: {df_carencias.shape}")
        print(f"  Intervenciones Persona: {df_intervenciones.shape}")
        
        print(f"\n🏷️  Hogares únicos detectados:")
        hogares_originales = len(df_hogares)
        print(f"  - En CaracteristicasHogar: {hogares_originales}")
        print(f"  - En CaracteristicasPersona: {df_caracteristicas['id_hogar'].nunique()}")
        
        # Identificar hogares huérfanos
        print(f"\n🔎 Identificando hogares huérfanos...")
        hogares_validos = set(df_hogares['id_hogar'])
        hogares_en_personas = set(df_caracteristicas['id_hogar'])
        hogares_huerfanos = hogares_en_personas - hogares_validos
        
        if len(hogares_huerfanos) > 0:
            print(f"  ⚠️  Hogares huérfanos encontrados: {len(hogares_huerfanos)}")
            df_huerfanos_temp = df_caracteristicas[df_caracteristicas['id_hogar'].isin(hogares_huerfanos)]
            print(f"     Personas en hogares huérfanos: {len(df_huerfanos_temp):,}")
            print(f"     IDs (primeros 10): {sorted(list(hogares_huerfanos))[:10]}")
            
            df_caracteristicas = df_caracteristicas[df_caracteristicas['id_hogar'].isin(hogares_validos)]
            print(f"  ✓ Características Persona después de filtro: {len(df_caracteristicas):,}")
            
            self._generar_reporte_hogares_huerfanos(df_huerfanos_temp, hogares_huerfanos)
        else:
            print(f"  ✓ No hay hogares huérfanos - todos válidos")
        
        print(f"\n🔗 Fusionando datasets de personas...")
        df_personas_completo = df_caracteristicas.merge(
            df_carencias, on=['id_hogar', 'id_persona'], how='inner'
        ).merge(
            df_intervenciones, on=['id_hogar', 'id_persona'], how='inner'
        )
        
        print(f"  ✓ Registros después de merges: {len(df_personas_completo):,}")
        print(f"  ✓ Hogares únicos: {df_personas_completo['id_hogar'].nunique():,}")
        
        print(f"\n🔗 Fusionando con características de hogar...")
        df_completo = df_personas_completo.merge(df_hogares, on='id_hogar', how='left')
        
        print(f"  ✓ Registros antes de limpieza: {len(df_completo):,}")
        
        print(f"\n🧹 Limpiando edades inválidas...")
        edades_invalidas = len(df_completo[(df_completo['edad_persona'] < 0) | (df_completo['edad_persona'] > 120)])
        
        df_completo = df_completo[
            (df_completo['edad_persona'] >= 0) & 
            (df_completo['edad_persona'] <= 120)
        ]
        
        print(f"  ✓ Registros eliminados: {edades_invalidas:,}")
        print(f"  ✓ Registros finales: {len(df_completo):,}")
        print(f"  ✓ Hogares únicos finales: {df_completo['id_hogar'].nunique():,}")
        
        print(f"\n✅ Dataset integrado y limpiado: {df_completo.shape}")
        
        self.df_completo = df_completo
        self.df_personas = df_personas_completo
        self.df_hogares = df_hogares
        
        return df_completo

    def _detectar_ruta_datos(self) -> str:
        """
        Detecta automáticamente la ruta correcta de 01_data/
        Funciona desde backend/ O desde frontend/ (Streamlit)
        """
        import os
        
        # Posibles rutas según dónde se ejecute el script
        rutas_posibles = [
            # Si se ejecuta desde backend/
            "../data/01_data/",
            # Si se ejecuta desde frontend/ (Streamlit)
            "../data/01_data/",
            "../../data/01_data/",
            # Si está en el mismo nivel que data/
            "./data/01_data/",
            "data/01_data/",
            # Búsqueda absoluta
            os.path.expanduser("~/analizador-politica-social/data/01_data/"),
        ]
        
        for ruta in rutas_posibles:
            ruta_abs = os.path.abspath(ruta)
            
            # Verificar que existe la carpeta Y contiene los archivos necesarios
            if os.path.exists(ruta_abs):
                archivos_esperados = [
                    "CaracteristicasHogar.csv",
                    "CaracteristicasPersona.csv",
                    "CarenciasPersona.csv",
                    "IntervencionesPotencialesPAPEPersona.csv"
                ]
                
                archivos_presentes = os.listdir(ruta_abs)
                if all(archivo in archivos_presentes for archivo in archivos_esperados):
                    print(f"   ✅ Ruta válida encontrada: {ruta_abs}")
                    return ruta
        
        # Si no encuentra nada, error informativo
        raise FileNotFoundError(
            f"\n❌ NO SE ENCONTRÓ LA CARPETA 01_data\n"
            f"   Directorio actual: {os.getcwd()}\n"
            f"   Rutas buscadas:\n"
            + "\n".join([f"      - {os.path.abspath(r)}" for r in rutas_posibles]) +
            f"\n\n   Soluciones:\n"
            f"   1. Verifica que la estructura sea:\n"
            f"      analizador-politica-social/\n"
            f"      ├── backend/\n"
            f"      ├── frontend/\n"
            f"      └── data/\n"
            f"          └── 01_data/\n"
            f"\n   2. O pasa la ruta manualmente:\n"
            f"      integrator.cargar_y_unir_datasets('../data/01_data/')"
        )


    def _generar_reporte_hogares_huerfanos(self, df_huerfanos: pd.DataFrame, ids_huerfanos: set):
        """Genera reportes de hogares huérfanos"""
        import os
        from datetime import datetime
        
        directorio_reportes = "05_reportes_datos"
        if not os.path.exists(directorio_reportes):
            os.makedirs(directorio_reportes)
            print(f"  📁 Directorio de reportes creado: {directorio_reportes}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Reporte 1: IDs
        archivo_ids = f"{directorio_reportes}/hogares_huerfanos_ids_{timestamp}.csv"
        df_ids = pd.DataFrame({'id_hogar_huerfano': sorted(list(ids_huerfanos))})
        df_ids.to_csv(archivo_ids, index=False)
        print(f"  📄 Reporte 1: {archivo_ids}")
        
        # Reporte 2: Personas
        archivo_personas = f"{directorio_reportes}/personas_en_hogares_huerfanos_{timestamp}.csv"
        columnas_reales = [col for col in ['id_hogar', 'id_persona', 'edad_persona', 'sexo_persona', 'parentesco_persona', 'tipo_persona'] 
                          if col in df_huerfanos.columns]
        df_personas = df_huerfanos[columnas_reales].copy().sort_values('id_hogar')
        df_personas.to_csv(archivo_personas, index=False)
        print(f"  📄 Reporte 2: {archivo_personas}")
        
        # Reporte 3: Estadísticas
        archivo_stats = f"{directorio_reportes}/estadisticas_hogares_huerfanos_{timestamp}.csv"
        stats = df_huerfanos.groupby('id_hogar').agg({
            'id_persona': 'count',
            'edad_persona': ['min', 'max', 'mean'],
            'sexo_persona': lambda x: (x == 'Mujer').sum()
        }).round(2)
        stats.columns = ['total_personas', 'edad_minima', 'edad_maxima', 'edad_promedio', 'cantidad_mujeres']
        stats = stats.reset_index()
        stats.to_csv(archivo_stats, index=False)
        print(f"  📄 Reporte 3: {archivo_stats}")
        
        # Reporte 4: Resumen
        archivo_resumen = f"{directorio_reportes}/resumen_hogares_huerfanos_{timestamp}.txt"
        cantidad_mujeres = (df_huerfanos['sexo_persona'] == 'Mujer').sum()
        cantidad_hombres = (df_huerfanos['sexo_persona'] == 'Hombre').sum()
        
        resumen = f"""
================================================================================
REPORTE DE HOGARES HUÉRFANOS
================================================================================
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RESUMEN
================================================================================
Hogares huérfanos encontrados: {len(ids_huerfanos):,}
Personas en hogares huérfanos: {len(df_huerfanos):,}
Porcentaje de personas: {(len(df_huerfanos) / 32723 * 100):.2f}%

DEFINICIÓN
Un hogar "huérfano" existe en CaracteristicasPersona.csv pero NO en 
CaracteristicasHogar.csv. Fueron eliminados por integridad de datos.

DISTRIBUCIÓN POR SEXO
Mujeres: {cantidad_mujeres:,} ({(cantidad_mujeres / len(df_huerfanos) * 100):.1f}%)
Hombres: {cantidad_hombres:,} ({(cantidad_hombres / len(df_huerfanos) * 100):.1f}%)

DISTRIBUCIÓN POR EDAD
Edad mínima: {df_huerfanos['edad_persona'].min():.0f} años
Edad máxima: {df_huerfanos['edad_persona'].max():.0f} años
Edad promedio: {df_huerfanos['edad_persona'].mean():.1f} años

ARCHIVOS GENERADOS
1. hogares_huerfanos_ids_{timestamp}.csv
2. personas_en_hogares_huerfanos_{timestamp}.csv
3. estadisticas_hogares_huerfanos_{timestamp}.csv
4. resumen_hogares_huerfanos_{timestamp}.txt

================================================================================
"""
        
        with open(archivo_resumen, 'w', encoding='utf-8') as f:
            f.write(resumen)
        print(f"  📄 Reporte 4: {archivo_resumen}")
        print(f"\n  ✅ Reportes generados en: {directorio_reportes}/")

    def auditar_integridad_completa(self):
        """Auditoría exhaustiva del dataset"""
        print("\n" + "="*70)
        print("🔍 AUDITORÍA COMPLETA DE INTEGRIDAD DE DATOS")
        print("="*70)
        
        if self.df_completo is None:
            print("❌ Error: Dataset no cargado")
            return
        
        total_personas = len(self.df_completo)
        total_hogares = self.df_completo['id_hogar'].nunique()
        
        print(f"\n📊 MÉTRICAS PRINCIPALES:")
        print(f"  • Personas: {total_personas:,}")
        print(f"  • Hogares: {total_hogares:,}")
        print(f"  • Colonias: {self.df_completo['colonia'].nunique():,}")
        print(f"  • AGEBs: {self.df_completo['ageb'].nunique():,}")
        
        personas_por_hogar = self.df_completo.groupby('id_hogar').size()
        print(f"\n👥 PERSONAS POR HOGAR:")
        print(f"  • Promedio: {personas_por_hogar.mean():.2f}")
        print(f"  • Mediana: {personas_por_hogar.median():.0f}")
        
        hogares_originales = len(self.df_hogares)
        print(f"\n✅ COHERENCIA:")
        print(f"  • Hogares originales: {hogares_originales:,}")
        print(f"  • Hogares finales: {total_hogares:,}")
        
        if total_hogares == hogares_originales:
            print(f"  ✅ INTEGRIDAD PERFECTA")
        else:
            print(f"  ⚠️  Diferencia: {abs(hogares_originales - total_hogares):,}")
        
        print(f"\n{'='*70}")
        print(f"✅ AUDITORÍA COMPLETADA")
        print(f"{'='*70}\n")
        
        return {
            'total_personas': total_personas,
            'total_hogares': total_hogares,
            'personas_por_hogar_promedio': personas_por_hogar.mean(),
            'integridad_hogares': total_hogares == hogares_originales
        }

    def comparar_reportes_integridad(self, directorio_reportes: str = "05_reportes_datos"):
        """Compara reportes de diferentes ejecuciones"""
        import os
        import glob
        
        print("\n" + "="*70)
        print("📊 COMPARADOR DE REPORTES DE INTEGRIDAD")
        print("="*70)
        
        if not os.path.exists(directorio_reportes):
            print(f"⚠️  No hay reportes previos para comparar")
            return None
        
        archivos_ids = sorted(glob.glob(f"{directorio_reportes}/hogares_huerfanos_ids_*.csv"))
        
        if len(archivos_ids) < 2:
            print(f"⚠️  Se necesitan 2+ reportes para comparar (encontrados: {len(archivos_ids)})")
            return None
        
        archivo_anterior = archivos_ids[-2]
        archivo_actual = archivos_ids[-1]
        
        print(f"\n📁 REPORTES A COMPARAR:")
        print(f"  Anterior: {os.path.basename(archivo_anterior)}")
        print(f"  Actual:   {os.path.basename(archivo_actual)}")
        
        df_anterior = pd.read_csv(archivo_anterior)
        df_actual = pd.read_csv(archivo_actual)
        
        ids_anterior = set(df_anterior['id_hogar_huerfano'].astype(int))
        ids_actual = set(df_actual['id_hogar_huerfano'].astype(int))
        
        nuevos = ids_actual - ids_anterior
        recuperados = ids_anterior - ids_actual
        
        print(f"\n📈 ANÁLISIS:")
        print(f"  • Anteriores: {len(ids_anterior):,}")
        print(f"  • Actuales: {len(ids_actual):,}")
        print(f"  • Nuevos: {len(nuevos):,}")
        print(f"  • Recuperados: {len(recuperados):,}")
        print(f"  • Cambio neto: {len(ids_actual) - len(ids_anterior):+d}")
        
        print(f"\n{'='*70}\n")
        
        return {
            'nuevos': sorted(list(nuevos)),
            'recuperados': sorted(list(recuperados)),
            'cambio_neto': len(ids_actual) - len(ids_anterior)
        }

    def generar_reporte_temporal(self, directorio_reportes: str = "05_reportes_datos"):
        """Genera timeline de integridad"""
        import os
        import glob
        
        print("\n📊 REPORTE TEMPORAL DE INTEGRIDAD")
        
        if not os.path.exists(directorio_reportes):
            return None
        
        archivos = sorted(glob.glob(f"{directorio_reportes}/hogares_huerfanos_ids_*.csv"))
        
        if len(archivos) == 0:
            print("⚠️  No hay reportes")
            return None
        
        print(f"Reportes encontrados: {len(archivos)}\n")
        
        for i, archivo in enumerate(archivos, 1):
            df = pd.read_csv(archivo)
            timestamp = os.path.basename(archivo).split('_')[-1].replace('.csv', '')
            print(f"{i}. {timestamp}: {len(df):,} hogares huérfanos")
        
        return True

    def limpiar_reportes_antiguos(self, directorio_reportes: str = "05_reportes_datos", dias_retencion: int = 30):
        """Limpia reportes antiguos"""
        import os
        import glob
        import time
        
        print(f"\n🧹 Limpiando reportes más antiguos de {dias_retencion} días...")
        
        if not os.path.exists(directorio_reportes):
            return
        
        archivo_actual = time.time()
        segundos = dias_retencion * 24 * 60 * 60
        eliminados = 0
        
        for archivo in glob.glob(f"{directorio_reportes}/*_*.csv") + glob.glob(f"{directorio_reportes}/*_*.txt"):
            if os.path.getmtime(archivo) < archivo_actual - segundos:
                try:
                    os.remove(archivo)
                    eliminados += 1
                except:
                    pass
        
        print(f"✅ {eliminados} archivos eliminados")


# ============================================================================
# 2. ARQUITECTURA MODULAR - NUEVAS CLASES ESPECIALIZADAS
# ============================================================================

class DelimitadorPoblacional:
    """Se especializa SOLO en delimitar segmentos poblacionales"""
    
    def __init__(self, df_completo: pd.DataFrame):
        self.df = df_completo
    
    @staticmethod
    def validar_rango_edad(edad_min: int, edad_max: int) -> Tuple[int, int]:
        """
        Valida y corrige rangos de edad absurdos.
        
        Reglas:
        - edad_min debe estar en [0, 150]
        - edad_max debe estar en [0, 150]
        - edad_min <= edad_max
        - Si está invertido, intercambia automáticamente
        
        Returns:
            Tuple[int, int]: (edad_min_validada, edad_max_validada)
            
        Raises:
            ValueError: Si rango es completamente inválido
        """
        EDAD_MIN_VALIDA = 0
        EDAD_MAX_VALIDA = 150
        
        try:
            edad_min = int(edad_min)
            edad_max = int(edad_max)
        except (ValueError, TypeError):
            raise ValueError(f"Edades deben ser números")
        
        # Advertencia: valores muy altos
        if edad_min > 120 or edad_max > 120:
            print(f"⚠️  [VALIDACIÓN] Edad inusualmente alta: [{edad_min}, {edad_max}]")
        
        # Corregir valores negativos
        if edad_min < 0:
            print(f"⚠️  [VALIDACIÓN] Edad negativa {edad_min} corregida a 0")
            edad_min = 0
        
        if edad_max < 0:
            raise ValueError(f"Edad máxima no puede ser negativa: {edad_max}")
        
        # Corregir valores fuera de rango
        if edad_min > EDAD_MAX_VALIDA:
            raise ValueError(f"Edad mínima {edad_min} excede máximo válido")
        
        if edad_max > EDAD_MAX_VALIDA:
            print(f"⚠️  [VALIDACIÓN] Edad máxima {edad_max} ajustada a {EDAD_MAX_VALIDA}")
            edad_max = EDAD_MAX_VALIDA
        
        # Intercambiar si está invertido
        if edad_min > edad_max:
            print(f"⚠️  [VALIDACIÓN] Rango invertido [{edad_min}, {edad_max}] intercambiado")
            edad_min, edad_max = edad_max, edad_min
        
        return (edad_min, edad_max)

    def aplicar_filtros(self, criterios: Dict) -> pd.DataFrame:
        """Aplica filtros demográficos y retorna DataFrame del segmento"""
        df_filtrado = self.df.copy()
        condiciones = []

        # Criterios de edad
        if 'rango_edad' in criterios and criterios['rango_edad']:
            try:
                edad_min, edad_max = self.validar_rango_edad(
                    criterios['rango_edad'][0], 
                    criterios['rango_edad'][1]
                )
                condiciones.append(
                    (df_filtrado['edad_persona'] >= edad_min) & 
                    (df_filtrado['edad_persona'] <= edad_max)
                )
            except ValueError as e:
                print(f"❌ [RANGO EDAD] Error: {e}")
                return df_filtrado
                   
        # Criterios de sexo
        if 'sexo' in criterios and criterios['sexo']:
            sexo_map = {'Mujer': 'Mujer', 'Hombre': 'Hombre', 'M': 'Mujer', 'H': 'Hombre'}
            sexo_valor = sexo_map.get(criterios['sexo'], criterios['sexo'])
            condiciones.append(df_filtrado['sexo_persona'] == sexo_valor)
        
        # Criterios de ubicación
        if 'ubicacion' in criterios and criterios['ubicacion']:
            mascara_ubicacion = (
                df_filtrado['colonia'].str.contains(criterios['ubicacion'], case=False, na=False) |
                df_filtrado['ageb'].str.contains(criterios['ubicacion'], case=False, na=False) |
                df_filtrado['ubicacion'].str.contains(criterios['ubicacion'], case=False, na=False)
            )
            condiciones.append(mascara_ubicacion)
        
        # Criterios de carencias
        carencia_map = {
            'carencia_salud': 'presencia_carencia_salud_persona',
            'carencia_educacion': 'presencia_rezago_educativo_persona',
            'carencia_seguridad_social': 'presencia_carencia_seguridad_social_persona'
        }
        
        for carencia_key, carencia_columna in carencia_map.items():
            if carencia_key in criterios and criterios[carencia_key]:
                condiciones.append(df_filtrado[carencia_columna] == 'yes')
        
        # Criterios de programas
        if 'programa_social' in criterios and criterios['programa_social']:
            programa_columna = f"es_elegible_{criterios['programa_social']}"
            if programa_columna in self.df.columns:
                condiciones.append(df_filtrado[programa_columna] == 'yes')
        
        # Aplicar todos los filtros
        if condiciones:
            mascara_final = condiciones[0]
            for cond in condiciones[1:]:
                mascara_final &= cond
            df_filtrado = df_filtrado[mascara_final]
        
        return df_filtrado
class AnalizadorDemografico:
    """Se especializa SOLO en análisis demográfico de segmentos"""
    
    def __init__(self, df_completo: pd.DataFrame):
        self.df = df_completo
    
    def generar_perfil_segmento(self, df_segmento: pd.DataFrame) -> Dict:
        """Genera perfil demográfico del segmento - VERSIÓN CORREGIDA JSON"""
        if len(df_segmento) == 0:
            return {}
        
        # Corrección: Asegurar tipos nativos de Python para JSON
        edad_prom = df_segmento['edad_persona'].mean()
        personas_hogar_prom = df_segmento.groupby('id_hogar').size().mean()
        
        distrib_sexo = df_segmento['sexo_persona'].value_counts().to_dict()
        # Convertir valores numpy.int64 a int nativo
        distrib_sexo_nativo = {str(k): int(v) for k, v in distrib_sexo.items()}

        return {
            "total_personas": int(len(df_segmento)),
            "edad_promedio": float(round(edad_prom, 1)) if pd.notna(edad_prom) else 0.0,
            "distribucion_sexo": distrib_sexo_nativo,
            "hogares_afectados": int(df_segmento['id_hogar'].nunique()),
            "personas_por_hogar": float(round(personas_hogar_prom, 2)) if pd.notna(personas_hogar_prom) else 0.0
        }
    
    def analizar_distribucion_geografica(self, df_segmento: pd.DataFrame, top_n: int = 3) -> Dict:
        """Analiza distribución geográfica del segmento"""
        if len(df_segmento) == 0:
            return {}
            
        ranking_colonias = df_segmento['colonia'].value_counts().head(top_n)
        porcentajes_colonias = (ranking_colonias / len(df_segmento) * 100).round(2)
        
        return {
            "top_colonias": {
                "nombres": ranking_colonias.index.tolist(),
                "conteos": ranking_colonias.values.tolist(),
                "porcentajes": porcentajes_colonias.values.tolist()
            },
            "total_colonias_afectadas": df_segmento['colonia'].nunique()
        }
class AnalizadorProgramasSociales:
    """
    Clase para analizar elegibilidad y perfil de beneficiarios potenciales
    de programas sociales, con capacidad de segmentación geográfica y por carencias.
    """
    
    def __init__(self, df_completo: pd.DataFrame, api_key:str):
        """Inicializa el analizador de programas sociales"""
        self.df = df_completo
        self.api_key=api_key
        
        # Detectar automáticamente los programas disponibles
        self.programas_disponibles = [col for col in self.df.columns if col.startswith('es_elegible_')]
        self.programas_nombres = [col.replace('es_elegible_', '') for col in self.programas_disponibles]
        
        # Mapeo mejorado de programas
        self.mapeo_programas = {
            'imss_bienestar': 'IMSS Bienestar',
            'pension_adultos_mayores': 'Pensión Adultos Mayores', 
            'pension_mujeres_bienestar': 'Pensión Mujeres Bienestar',
            'beca_benito_juarez': 'Beca Benito Juárez',
            'beca_rita_cetina': 'Beca Rita Cetina',
            'jovenes_escribiendo_el_futuro': 'Jóvenes Escribiendo el Futuro',
            'jovenes_construyendo_futuro': 'Jóvenes Construyendo el Futuro',
            'desde_la_cuna': 'Desde la Cuna',
            'mi_beca_para_empezar': 'Mi Beca para Empezar',
            'seguro_desempleo_cdmx': 'Seguro de Desempleo CDMX',
            'ingreso_ciudadano_universal': 'Ingreso Ciudadano Universal',
            'inea': 'INEA',
            'leche_bienestar': 'Leche Bienestar'
        }

    # ========================================================================
    # MÉTODOS AUXILIARES DE FILTRADO
    # ========================================================================
    
    def _obtener_programas_por_carencia(self, carencia: str) -> List[str]:
        """Mapeo mejorado de programas relevantes por tipo de carencia"""
        mapeo_programas_carencias = {
            'salud': ['imss_bienestar', 'seguro_desempleo_cdmx', 'pension_adultos_mayores'],
            'educacion': ['mi_beca_para_empezar', 'beca_rita_cetina', 'beca_benito_juarez', 
                         'jovenes_escribiendo_el_futuro', 'inea', 'desde_la_cuna'],
            'seguridad_social': ['pension_adultos_mayores', 'pension_mujeres_bienestar', 
                               'ingreso_ciudadano_universal', 'seguro_desempleo_cdmx', 'imss_bienestar']
        }
        return mapeo_programas_carencias.get(carencia, [])

    # ***** CORREGIDO *****
    def _aplicar_filtros_basicos(self, rango_edad: tuple = None, ubicacion: str = None, 
                               sexo: str = None, carencia: str = None,parentesco: str = None) -> pd.DataFrame:
        """Aplica filtros básicos de forma consistente"""
        df_filtrado = self.df.copy()
        
        # Filtro por edad
        if rango_edad:
            edad_min, edad_max = rango_edad
            df_filtrado = df_filtrado[
                (df_filtrado['edad_persona'] >= edad_min) & 
                (df_filtrado['edad_persona'] <= edad_max)
            ]
        
        # Filtro por sexo
        if sexo:
            sexo_map = {'Mujer': 'Mujer', 'Hombre': 'Hombre', 'M': 'Mujer', 'H': 'Hombre'}
            sexo_valor = sexo_map.get(sexo, sexo)
            df_filtrado = df_filtrado[df_filtrado['sexo_persona'] == sexo_valor]
        
        # Filtro por ubicación
        if ubicacion:
            mascara_ubicacion = (
                df_filtrado['colonia'].str.contains(ubicacion, case=False, na=False) |
                df_filtrado['ageb'].str.contains(ubicacion, case=False, na=False) |
                df_filtrado['ubicacion'].str.contains(ubicacion, case=False, na=False)
            )
            df_filtrado = df_filtrado[mascara_ubicacion]
        
        # Filtro por carencia
        if carencia:
            carencia_map = {
                'salud': 'presencia_carencia_salud_persona',
                'educacion': 'presencia_rezago_educativo_persona',
                'seguridad_social': 'presencia_carencia_seguridad_social_persona'
            }
            if carencia in carencia_map:
                df_filtrado = df_filtrado[df_filtrado[carencia_map[carencia]] == 'yes']

        # ✓ FILTRO DE PARENTESCO (CORRECCIÓN #4)
        if parentesco:
            df_filtrado= df_filtrado[df_filtrado['parentesco_persona'] == parentesco]
            print(f"   ↳ Parentesco: {parentesco}")

        return df_filtrado

    # ========================================================================
    # MÉTODO PRINCIPAL: ANÁLISIS DE ELEGIBILIDAD
    # ========================================================================
    
    # ***** CORREGIDO *****
    def analizar_elegibilidad_programa(self, programa: str, rango_edad: tuple = None, 
                                     ubicacion: str = None, sexo: str = None,
                                     carencia: str = None, incluir_brecha: bool = True,
                                     segmentacion_geografica: str = None) -> Dict:
        """
        Análisis de elegibilidad para un programa específico
        Versión final: Solo 5 secciones clave
        """
        columna_programa = f"es_elegible_{programa}"
        
        if columna_programa not in self.df.columns:
            return {
                "error": f"Programa '{programa}' no encontrado", 
                "programas_disponibles": self.programas_nombres,
                "columnas_reales": self.programas_disponibles
            }
        
        # Aplicar filtros
        df_filtrado = self._aplicar_filtros_basicos(rango_edad=rango_edad, ubicacion=ubicacion, sexo=sexo, carencia=carencia,parentesco=criterios.get("parentesco"))
        
        if len(df_filtrado) == 0:
            return {"error": "No hay personas que cumplan los criterios especificados"}
        
        # Personas elegibles
        df_elegibles = df_filtrado[df_filtrado[columna_programa] == 'yes']
        total_elegibles = len(df_elegibles)
        total_poblacion = len(df_filtrado)
        
        # ====================================================================
        # SECCIÓN 1: POBLACIÓN ELEGIBLE
        # ====================================================================
        poblacion_elegible = {
            "total_elegibles": int(total_elegibles),
            "tasa_elegibilidad": float(round((total_elegibles / total_poblacion * 100), 2)) if total_poblacion > 0 else 0.0,
            "hogares_afectados": int(df_elegibles['id_hogar'].nunique()) if total_elegibles > 0 else 0,
            "porcentaje_del_total_estudio": float(round((total_elegibles / len(self.df) * 100), 2)) if len(self.df) > 0 else 0.0
        }
        
        # ====================================================================
        # SECCIÓN 2: PERFIL DE LOS ELEGIBLES
        # ====================================================================
        perfil_elegibles = {
            "edad_promedio": float(round(df_elegibles['edad_persona'].mean(), 1)) if total_elegibles > 0 else 0.0,
            "edad_minima": int(df_elegibles['edad_persona'].min()) if total_elegibles > 0 else 0,
            "edad_maxima": int(df_elegibles['edad_persona'].max()) if total_elegibles > 0 else 0,
            "distribucion_sexo": {
                str(k): int(v) for k, v in 
                df_elegibles['sexo_persona'].value_counts().items()
            } if total_elegibles > 0 else {},
            "distribucion_sexo_porcentaje": {
                str(k): float(round((v / total_elegibles * 100), 1)) 
                for k, v in df_elegibles['sexo_persona'].value_counts().items()
            } if total_elegibles > 0 else {}
        }
        
        # ====================================================================
        # SECCIÓN 3: CARENCIAS DETECTADAS ENTRE ELEGIBLES
        # ====================================================================
        carencias = [
            ('presencia_carencia_salud_persona', 'Salud'),
            ('presencia_rezago_educativo_persona', 'Educación'),
            ('presencia_carencia_seguridad_social_persona', 'Seguridad Social')
        ]
        
        carencias_detectadas = {}
        for columna_carencia, nombre_carencia in carencias:
            if columna_carencia in df_elegibles.columns:
                total_con_carencia = len(df_elegibles[df_elegibles[columna_carencia] == 'yes'])
                porcentaje = (total_con_carencia / total_elegibles * 100) if total_elegibles > 0 else 0
                
                carencias_detectadas[nombre_carencia] = {
                    "cantidad": int(total_con_carencia),
                    "porcentaje": float(round(porcentaje, 2))
                }
        
        # ====================================================================
        # SECCIÓN 4: CONTEXTO DE LA COLONIA
        # ====================================================================
        contexto_colonia = {}
        if ubicacion:
            contexto_colonia = self._obtener_contexto_colonia(ubicacion, df_elegibles)
        
        # ====================================================================
        # SECCIÓN 5: COMPARATIVA INTERCOLONIAL
        # ====================================================================
        comparativa = {}
        if ubicacion:
            comparativa = self._generar_comparativa(programa, ubicacion, df_filtrado)
        
        # ====================================================================
        # RESULTADO FINAL
        # ====================================================================
        resultado = {
            "programa": programa,
            "nombre_programa": self.mapeo_programas.get(programa, programa),
            "criterios_aplicados": {
                "rango_edad": rango_edad,
                "ubicacion": ubicacion,
                "sexo": sexo,
                "carencia": carencia
            },
            "seccion_1_poblacion_elegible": poblacion_elegible,
            "seccion_2_perfil_elegibles": perfil_elegibles,
            "seccion_3_carencias_detectadas": carencias_detectadas,
            "seccion_4_contexto_colonia": contexto_colonia,
            "seccion_5_comparativa": comparativa
        }
        
        return resultado

    # ========================================================================
    # MÉTODOS AUXILIARES PARA SECCIONES
    # ========================================================================
    
    def _obtener_contexto_colonia(self, ubicacion: str, df_elegibles: pd.DataFrame) -> Dict:
        """Obtiene el contexto de la colonia"""
        df_colonia = self.df[
            (self.df['colonia'].str.contains(ubicacion, case=False, na=False)) |
            (self.df['ageb'].str.contains(ubicacion, case=False, na=False)) |
            (self.df['ubicacion'].str.contains(ubicacion, case=False, na=False))
        ]
        
        total_colonia = len(df_colonia)
        hogares_colonia = df_colonia['id_hogar'].nunique()
        edad_promedio_colonia = df_colonia['edad_persona'].mean()
        
        contexto = {
            "poblacion_total": int(total_colonia),
            "hogares_totales": int(hogares_colonia),
            "edad_promedio": float(round(edad_promedio_colonia, 1)),
            "ranking_colonias": self._obtener_ranking_colonia(ubicacion),
            "caracteristicas": self._caracterizar_colonia(df_colonia, df_elegibles)
        }
        
        return contexto

    def _obtener_ranking_colonia(self, ubicacion: str) -> Dict:
        """Obtiene ranking de la colonia en términos de población"""
        colonias_ranking = self.df['colonia'].value_counts()
        
        try:
            posicion = list(colonias_ranking.index).index(ubicacion) + 1
            total_colonias = len(colonias_ranking)
            poblacion_colonia = colonias_ranking[ubicacion]
            
            return {
                "posicion": int(posicion),
                "de_total": int(total_colonias),
                "poblacion": int(poblacion_colonia)
            }
        except:
            return {}

    def _caracterizar_colonia(self, df_colonia: pd.DataFrame, df_elegibles: pd.DataFrame) -> Dict:
        """Caracteriza la colonia en términos de vulnerabilidad"""
        caracteristicas = {}
        
        if len(df_colonia) > 0:
            carencias_col = {
                'salud': int(len(df_colonia[df_colonia['presencia_carencia_salud_persona'] == 'yes'])),
                'educacion': int(len(df_colonia[df_colonia['presencia_rezago_educativo_persona'] == 'yes'])),
                'seguridad_social': int(len(df_colonia[df_colonia['presencia_carencia_seguridad_social_persona'] == 'yes']))
            }
            
            caracteristicas = {
                "nivel_vulnerabilidad": self._clasificar_vulnerabilidad(carencias_col, len(df_colonia)),
                "carencias_principales": carencias_col
            }
        
        return caracteristicas

    def _clasificar_vulnerabilidad(self, carencias: Dict, total: int) -> str:
        """Clasifica nivel de vulnerabilidad"""
        if total == 0:
            return "Sin datos"
        
        promedio_carencias = sum(carencias.values()) / len(carencias) / total * 100
        
        if promedio_carencias > 60:
            return "Muy Alta"
        elif promedio_carencias > 45:
            return "Alta"
        elif promedio_carencias > 30:
            return "Moderada"
        else:
            return "Baja"

    def _generar_comparativa(self, programa: str, ubicacion: str, df_filtrado: pd.DataFrame) -> Dict:
        """Genera comparativa con otras colonias"""
        columna_programa = f"es_elegible_{programa}"
        
        # Top 5 colonias por elegibilidad del programa
        elegibilidad_por_colonia = df_filtrado.groupby('colonia').apply(
            lambda x: {
                'elegibles': len(x[x[columna_programa] == 'yes']),
                'total': len(x)
            }
        )
        
        # Calcular tasa
        comparativa_data = []
        for colonia, datos in elegibilidad_por_colonia.items():
            if datos['total'] > 0:
                tasa = (datos['elegibles'] / datos['total'] * 100)
                comparativa_data.append({
                    'colonia': str(colonia),
                    'elegibles': int(datos['elegibles']),
                    'tasa': float(round(tasa, 1))
                })
        
        # Ordenar por elegibles
        comparativa_data = sorted(comparativa_data, key=lambda x: x['elegibles'], reverse=True)
        
        # Encontrar posición de la colonia actual
        posicion_actual = next(
            (i+1 for i, x in enumerate(comparativa_data) if x['colonia'].lower() == ubicacion.lower()),
            None
        )
        
        return {
            "top_colonias": comparativa_data[:5],
            "posicion_colonia_actual": int(posicion_actual) if posicion_actual else None,
            "total_colonias_comparadas": int(len(comparativa_data))
        }

    # ========================================================================
    # MÉTODOS COMPLEMENTARIOS
    # ========================================================================
    
    # ***** CORREGIDO *****
    def analizar_elegibilidad_multiple(self, programas: List[str], rango_edad: tuple = None,
                                     ubicacion: str = None, top_n: int = 5) -> Dict:
        """Analiza elegibilidad para múltiples programas simultáneamente"""
        resultados = {}
        comparativa = {}
        
        for programa in programas:
            resultado = self.analizar_elegibilidad_programa(
                programa=programa,
                rango_edad=rango_edad,
                ubicacion=ubicacion,
                incluir_brecha=True
            )
            
            if "error" not in resultado:
                resultados[programa] = resultado
                metricas = resultado.get('seccion_1_poblacion_elegible', {})
                comparativa[programa] = {
                    "total_elegibles": metricas.get("total_elegibles", 0),
                    "tasa_elegibilidad": metricas.get("tasa_elegibilidad", 0)
                }
        
        if comparativa:
            programas_ordenados = sorted(comparativa.keys(), 
                                       key=lambda x: comparativa[x]["total_elegibles"], 
                                       reverse=True)
            
            analisis_comparativo = {
                "programa_mayor_elegibilidad": programas_ordenados[0] if programas_ordenados else None,
                "ranking_elegibles": programas_ordenados[:top_n]
            }
        else:
            analisis_comparativo = {}
        
        return {
            "analisis_individual": resultados,
            "analisis_comparativo": analisis_comparativo,
            "resumen": {
                "total_programas_analizados": len(resultados),
                "total_elegibles_todos": sum(c.get("total_elegibles", 0) for c in comparativa.values())
            }
        }

    # ***** CORREGIDO *****
    def identificar_carencias_sin_cobertura(self, carencia: str, rango_edad: tuple = None, 
                                           ubicacion: str = None) -> Dict:
        """Personas con carencias que NO son elegibles para programas relacionados"""
        print(f"🔍 [CARENCIAS_SIN_COBERTURA] Iniciando análisis...")
        
        try:
            carencia_map = {
                'salud': 'presencia_carencia_salud_persona',
                'educacion': 'presencia_rezago_educativo_persona',
                'seguridad_social': 'presencia_carencia_seguridad_social_persona'
            }
            
            if carencia not in carencia_map:
                return {"error": f"Carencia '{carencia}' no reconocida. Válidas: {list(carencia_map.keys())}"}
            
            columna_carencia = carencia_map[carencia]
            
            if columna_carencia not in self.df.columns:
                return {"error": f"Columna de carencia no encontrada: {columna_carencia}"}
            
            # Filtrar población con la carencia
            df_con_carencia = self.df[self.df[columna_carencia] == 'yes'].copy()
            
            if len(df_con_carencia) == 0:
                return {"error": f"No hay personas con carencia de {carencia}"}
            
            # Aplicar filtros adicionales
            if rango_edad:
                edad_min, edad_max = rango_edad
                df_con_carencia = df_con_carencia[
                    (df_con_carencia['edad_persona'] >= edad_min) & 
                    (df_con_carencia['edad_persona'] <= edad_max)
                ]
            
            if ubicacion:
                mascara_ubicacion = (
                    df_con_carencia['colonia'].str.contains(ubicacion, case=False, na=False) |
                    df_con_carencia['ageb'].str.contains(ubicacion, case=False, na=False) |
                    df_con_carencia['ubicacion'].str.contains(ubicacion, case=False, na=False)
                )
                df_con_carencia = df_con_carencia[mascara_ubicacion]
            
            total_con_carencia = len(df_con_carencia)
            
            # Obtener programas relacionados
            programas_relacionados = self._obtener_programas_por_carencia(carencia)
            
            if not programas_relacionados:
                return {"error": f"No hay programas relacionados con carencia de {carencia}"}
            
            # Identificar personas sin elegibilidad para programas
            df_sin_cobertura = df_con_carencia.copy()
            
            mascara_elegible = None
            for programa in programas_relacionados:
                columna_programa = f"es_elegible_{programa}"
                if columna_programa in self.df.columns:
                    if mascara_elegible is None:
                        mascara_elegible = (df_con_carencia[columna_programa] == 'yes')
                    else:
                        mascara_elegible |= (df_con_carencia[columna_programa] == 'yes')
            
            if mascara_elegible is None:
                personas_sin_cobertura = df_con_carencia
            else:
                personas_sin_cobertura = df_con_carencia[~mascara_elegible]
            
            total_sin_cobertura = len(personas_sin_cobertura)
            tasa_brecha = (total_sin_cobertura / total_con_carencia * 100) if total_con_carencia > 0 else 0
            
            # Perfil de brecha
            perfil_brecha = {}
            if total_sin_cobertura > 0:
                perfil_brecha = {
                    "edad_promedio": float(round(personas_sin_cobertura['edad_persona'].mean(), 1)),
                    "edad_minima": int(personas_sin_cobertura['edad_persona'].min()),
                    "edad_maxima": int(personas_sin_cobertura['edad_persona'].max()),
                    "distribucion_sexo": {
                        str(k): int(v) for k, v in 
                        personas_sin_cobertura['sexo_persona'].value_counts().items()
                    },
                    "hogares_afectados": int(personas_sin_cobertura['id_hogar'].nunique())
                }
            
            resultado = {
                "tipo_analisis": "carencias_sin_cobertura",
                "carencia": carencia,
                "metricas_principales": {
                    "total_personas_con_carencia": int(total_con_carencia),
                    "total_personas_sin_cobertura": int(total_sin_cobertura),
                    "tasa_brecha": float(round(tasa_brecha, 2))
                },
                "programas_relacionados_analizados": programas_relacionados,
                "perfil_brecha": perfil_brecha
            }
            
            print(f"✅ [CARENCIAS_SIN_COBERTURA] Análisis completado exitosamente")
            return resultado
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"error": f"Error analizando carencias: {str(e)}"}

    # ***** CORREGIDO *****
    def analizar_intensidad_carencias(self, rango_edad: tuple = None, ubicacion: str = None) -> Dict:
        """Analiza personas con múltiples carencias simultáneas"""
        print(f"🔍 [INTENSIDAD_CARENCIAS] Iniciando análisis...")
        
        try:
            df_analisis = self.df.copy()
            
            if rango_edad:
                edad_min, edad_max = rango_edad
                df_analisis = df_analisis[
                    (df_analisis['edad_persona'] >= edad_min) & 
                    (df_analisis['edad_persona'] <= edad_max)
                ]
            
            if ubicacion:
                mascara_ubicacion = (
                    df_analisis['colonia'].str.contains(ubicacion, case=False, na=False) |
                    df_analisis['ageb'].str.contains(ubicacion, case=False, na=False) |
                    df_analisis['ubicacion'].str.contains(ubicacion, case=False, na=False)
                )
                df_analisis = df_analisis[mascara_ubicacion]
            
            columnas_carencias = [
                'presencia_carencia_salud_persona',
                'presencia_rezago_educativo_persona',
                'presencia_carencia_seguridad_social_persona'
            ]
            
            columnas_validas = [col for col in columnas_carencias if col in df_analisis.columns]
            
            df_analisis['total_carencias'] = 0
            for col in columnas_validas:
                df_analisis['total_carencias'] += (df_analisis[col] == 'yes').astype(int)
            
            distribucion_intensidad = df_analisis['total_carencias'].value_counts().sort_index()
            
            personas_sin_carencias = len(df_analisis[df_analisis['total_carencias'] == 0])
            personas_1_carencia = len(df_analisis[df_analisis['total_carencias'] == 1])
            personas_2_carencias = len(df_analisis[df_analisis['total_carencias'] == 2])
            personas_3_carencias = len(df_analisis[df_analisis['total_carencias'] == 3])
            
            total_personas = len(df_analisis)
            
            personas_vulnerabilidad_extrema = df_analisis[df_analisis['total_carencias'] == 3]
            
            resultado = {
                "tipo_analisis": "intensidad_carencias",
                "distribucion_intensidad": {
                    "sin_carencias": {
                        "cantidad": int(personas_sin_carencias),
                        "porcentaje": float(round((personas_sin_carencias / total_personas * 100), 2)) if total_personas > 0 else 0.0
                    },
                    "una_carencia": {
                        "cantidad": int(personas_1_carencia),
                        "porcentaje": float(round((personas_1_carencia / total_personas * 100), 2)) if total_personas > 0 else 0.0
                    },
                    "dos_carencias": {
                        "cantidad": int(personas_2_carencias),
                        "porcentaje": float(round((personas_2_carencias / total_personas * 100), 2)) if total_personas > 0 else 0.0
                    },
                    "tres_carencias": {
                        "cantidad": int(personas_3_carencias),
                        "porcentaje": float(round((personas_3_carencias / total_personas * 100), 2)) if total_personas > 0 else 0.0
                    }
                },
                "poblacion_vulnerabilidad_extrema": {
                    "total": int(len(personas_vulnerabilidad_extrema)),
                    "porcentaje": float(round((len(personas_vulnerabilidad_extrema) / total_personas * 100), 2)) if total_personas > 0 else 0.0
                }
            }
            
            print(f"✅ [INTENSIDAD_CARENCIAS] Análisis completado exitosamente")
            return resultado
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"error": f"Error analizando intensidad: {str(e)}"}

    # ***** CORREGIDO *****
    def analizar_brechas_multiprograma(self, programas: List[str], rango_edad: tuple = None,
                                      ubicacion: str = None) -> Dict:
        """Analiza y compara brechas entre múltiples programas"""
        print(f"🔍 [BRECHAS_MULTIPROGRAMA] Iniciando análisis comparativo...")
        
        try:
            resultados_programas = {}
            comparativa = {}
            
            for programa in programas:
                resultado = self.analizar_elegibilidad_programa(
                    programa=programa,
                    rango_edad=rango_edad,
                    ubicacion=ubicacion,
                    incluir_brecha=True
                )
                
                if "error" not in resultado:
                    resultados_programas[programa] = resultado
                    metricas = resultado.get('seccion_1_poblacion_elegible', {})
                    comparativa[programa] = {
                        "total_elegibles": metricas.get("total_elegibles", 0),
                        "tasa_elegibilidad": metricas.get("tasa_elegibilidad", 0)
                    }
            
            if comparativa:
                programas_ordenados = sorted(
                    comparativa.keys(),
                    key=lambda x: comparativa[x]["total_elegibles"],
                    reverse=True
                )
                
                analisis_comparativo = {
                    "programa_mas_cobertura": programas_ordenados[0] if programas_ordenados else None,
                    "ranking_por_elegibles": programas_ordenados
                }
            else:
                analisis_comparativo = {}
            
            resultado_final = {
                "tipo_analisis": "brechas_multiprograma",
                "resultados_por_programa": resultados_programas,
                "comparativa_resumida": comparativa,
                "analisis_comparativo": analisis_comparativo,
                "resumen_general": {
                    "total_programas_analizados": len(resultados_programas),
                    "total_elegibles_agregado": sum(c["total_elegibles"] for c in comparativa.values())
                }
            }
            
            print(f"✅ [BRECHAS_MULTIPROGRAMA] Análisis completado exitosamente")
            return resultado_final
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"error": f"Error analizando brechas: {str(e)}"}

    # ========================================================================
    # MÉTODOS FALTANTES IMPLEMENTADOS
    # ========================================================================

    # ***** CORREGIDO *****
    def analizar_brechas_programa_grupo(self, programa: str, rango_edad: tuple = None, 
                                        ubicacion: str = None) -> Dict:
        """
        ANÁLISIS DE BRECHAS: Personas elegibles que NO reciben un programa
        (Este método faltaba y fue llamado por el Agente LLM)
        """
        print(f"🔍 [BRECHA_PROGRAMA_GRUPO] Iniciando análisis para {programa}...")
        
        try:
            columna_programa = f"es_elegible_{programa}"
            
            if columna_programa not in self.df.columns:
                return {"error": f"Programa '{programa}' no encontrado"}

            # 1. Aplicar filtros base (edad, ubicación)
            df_filtrado = self._aplicar_filtros_basicos(rango_edad=rango_edad, ubicacion=ubicacion, parentesco=criterios.get("parentesco"))
            
            if len(df_filtrado) == 0:
                return {"error": "No hay personas que cumplan los criterios iniciales"}

            # 2. Identificar elegibles
            df_elegibles = df_filtrado[df_filtrado[columna_programa] == 'yes']
            total_elegibles = len(df_elegibles)
            
            if total_elegibles == 0:
                return {
                    "tipo_analisis": "brecha_programa",
                    "programa": programa,
                    "metricas_principales": {
                        "total_elegibles": 0,
                        "total_brecha": 0,
                        "tasa_brecha": 0.0
                    },
                    "perfil_brecha": {}
                }

            # 3. Identificar la brecha (Elegibles que NO reciben apoyos)
            # Asumimos que la columna 'recibe_apoyos_sociales' es el indicador
            if 'recibe_apoyos_sociales' not in df_elegibles.columns:
                 return {"error": "Columna 'recibe_apoyos_sociales' no encontrada para calcular la brecha"}
                 
            df_brecha = df_elegibles[df_elegibles['recibe_apoyos_sociales'] == 'no']
            total_brecha = len(df_brecha)
            
            tasa_brecha = (total_brecha / total_elegibles * 100) if total_elegibles > 0 else 0
            
            # 4. Perfilar la brecha
            perfil_brecha = {}
            if total_brecha > 0:
                perfil_brecha = {
                    "edad_promedio": float(round(df_brecha['edad_persona'].mean(), 1)),
                    "edad_minima": int(df_brecha['edad_persona'].min()),
                    "edad_maxima": int(df_brecha['edad_persona'].max()),
                    "distribucion_sexo": {
                        str(k): int(v) for k, v in 
                        df_brecha['sexo_persona'].value_counts().items()
                    },
                    "hogares_afectados": int(df_brecha['id_hogar'].nunique())
                }
            
            resultado = {
                "tipo_analisis": "brecha_programa",
                "programa": programa,
                "metricas_principales": {
                    "total_elegibles": int(total_elegibles),
                    "total_brecha": int(total_brecha),
                    "tasa_brecha": float(round(tasa_brecha, 2))
                },
                "perfil_brecha": perfil_brecha
            }
            
            print(f"✅ [BRECHA_PROGRAMA_GRUPO] Análisis completado")
            return resultado

        except Exception as e:
            print(f"❌ Error en analizar_brechas_programa_grupo: {str(e)}")
            return {"error": f"Error analizando brecha de programa: {str(e)}"}

    def analizar_cobertura_geografica(self, programa: str, nivel_geografico: str = "ageb", top_n: int = 10) -> Dict:
        """
        ANÁLISIS GEOGRÁFICO: Distribución de elegibles por AGEB o colonia
        (Este método faltaba y fue llamado por el Agente LLM)
        """
        print(f"🔍 [COBERTURA_GEO] Iniciando para {programa} por {nivel_geografico}...")
        
        try:
            columna_programa = f"es_elegible_{programa}"
            
            if columna_programa not in self.df.columns:
                return {"error": f"Programa '{programa}' no encontrado"}
            
            if nivel_geografico not in self.df.columns:
                return {"error": f"Nivel geográfico '{nivel_geografico}' no encontrado"}

            # 1. Obtener todos los elegibles para el programa
            df_elegibles = self.df[self.df[columna_programa] == 'yes']
            total_elegibles = len(df_elegibles)
            
            if total_elegibles == 0:
                return {"error": f"No hay personas elegibles para {programa}"}
            
            # 2. Agrupar por nivel geográfico
            conteo_geo = df_elegibles[nivel_geografico].value_counts().head(top_n)
            
            if len(conteo_geo) == 0:
                return {"error": f"No hay datos geográficos para {nivel_geografico}"}

            # 3. Calcular porcentajes (asegurando tipos nativos)
            porcentajes_geo = (conteo_geo / total_elegibles * 100).round(2)
            
            distribucion = []
            for zona in conteo_geo.index:
                distribucion.append({
                    "zona": str(zona),
                    "conteo": int(conteo_geo.loc[zona]),
                    "porcentaje": float(porcentajes_geo.loc[zona])
                })

            resultado = {
                "tipo_analisis": "cobertura_geografica",
                "programa": programa,
                "nivel_geografico": nivel_geografico,
                "total_elegibles_analizados": int(total_elegibles),
                "total_zonas_afectadas": int(df_elegibles[nivel_geografico].nunique()),
                "top_zonas": distribucion
            }
            
            print(f"✅ [COBERTURA_GEO] Análisis completado")
            return resultado

        except Exception as e:
            print(f"❌ Error en analizar_cobertura_geografica: {str(e)}")
            return {"error": f"Error analizando cobertura geo: {str(e)}"}   

# ============================================================================
# 3. COORDINADOR ANALÍTICO (MOVIDO ANTES DEL AGENTE PARA ORDEN CORRECTO)
# ============================================================================

class AnalizadorUnidimensional:
    """Coordina todos los módulos especializados - VERSIÓN COMPLETA CORREGIDA"""
    
    def __init__(self, df_completo: pd.DataFrame, api_key: str = None):
        self.df = df_completo
        self.delimitador = DelimitadorPoblacional(df_completo)
        self.demografico = AnalizadorDemografico(df_completo)
        self.programas = AnalizadorProgramasSociales(df_completo, api_key)
        self.esquema = self._generar_esquema_variables()
        self.generador_tablas = None # Se asignará después
    
    def _generar_esquema_variables(self) -> Dict[str, List[str]]:
        """Categoriza variables basado en la estructura real"""
        return {
            'categoricas': ['sexo_persona', 'parentesco_persona', 'tipo_persona', 'recibe_apoyos_sociales', 'ubicacion'],
            'numericas_discretas': ['edad_persona', 'personas', 'bebes', 'ancianos', 'total_personas'],
            'carencias': ['presencia_carencia_salud_persona', 'presencia_rezago_educativo_persona', 'presencia_carencia_seguridad_social_persona'],
            'programas': ['es_elegible_desde_la_cuna', 'es_elegible_mi_beca_para_empezar', 'es_elegible_beca_rita_cetina',
                         'es_elegible_beca_benito_juarez', 'es_elegible_jovenes_escribiendo_el_futuro', 
                         'es_elegible_jovenes_construyendo_futuro', 'es_elegible_seguro_desempleo_cdmx',
                         'es_elegible_ingreso_ciudadano_universal', 'es_elegible_pension_mujeres_bienestar',
                         'es_elegible_imss_bienestar', 'es_elegible_inea', 'es_elegible_leche_bienestar',
                         'es_elegible_pension_adultos_mayores'],
            'geograficas': ['colonia', 'ageb', 'manzana', 'ubicacion']
        }

    def _mapear_grupos_poblacionales(self) -> Dict[str, Any]:
        """MAPEO 100% PRECISO basado en estructura real de datos"""
        mapeo_grupos = {
            # ==================== GRUPOS POR EDAD ====================
            'niños': {'tipo': 'rango_edad', 'valor': [0, 12]},
            'niñas': {'tipo': 'rango_edad', 'valor': [0, 12]},
            'infantes': {'tipo': 'rango_edad', 'valor': [0, 3]},
            'bebés': {'tipo': 'rango_edad', 'valor': [0, 3]},
            'adolescentes': {'tipo': 'rango_edad', 'valor': [13, 18]},
            'jóvenes': {'tipo': 'rango_edad', 'valor': [19, 30]},
            'adultos': {'tipo': 'rango_edad', 'valor': [31, 50]},
            'adultos mayores': {'tipo': 'rango_edad', 'valor': [65, 100]},
            "mayores de 65": {'tipo': 'rango_edad', 'valor': [65, 100]},
            'tercera edad': {'tipo': 'rango_edad', 'valor': [65, 100]},
            'adulto mayor': {'tipo': 'rango_edad', 'valor': [65, 100]},
            
            # ==================== GRUPOS POR SEXO ====================
            'mujeres': {'tipo': 'sexo', 'valor': 'Mujer'},
            'hombres': {'tipo': 'sexo', 'valor': 'Hombre'},
            'mujer': {'tipo': 'sexo', 'valor': 'Mujer'},
            'hombre': {'tipo': 'sexo', 'valor': 'Hombre'},
            
            # ==================== CARENCIAS (VALORES REALES: 'yes'/'no') ====================
            'carencia de salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'carencia salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'sin salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'acceso a salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'carencia de acceso a salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'carencia acceso a salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'sin cobertura de salud': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            'sin servicios médicos': {'tipo': 'columna', 'valor': 'presencia_carencia_salud_persona', 'filtro': 'yes'},
            
            'carencia de educación': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'carencia educación': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'rezago educativo': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'educación': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'sin educación': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'sin asistir a la escuela': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'no asiste a la escuela': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},
            'inasistencia escolar': {'tipo': 'columna', 'valor': 'presencia_rezago_educativo_persona', 'filtro': 'yes'},


            'carencia de seguridad social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'carencia seguridad social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'seguridad social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'sin seguridad social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'carencia social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'sin afiliación': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'sin acceso a seguridad social': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},
            'no afiliadas': {'tipo': 'columna', 'valor': 'presencia_carencia_seguridad_social_persona', 'filtro': 'yes'},

            # ==================== CARENCIAS COMPLEJAS ====================
            'carencia máxima': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'múltiples carencias': {'tipo': 'multiple_carencias_min', 'valor': 2},  # ← 2 o más
            'carencia extrema': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'vulnerabilidad extrema': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'mayor carencia': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'pobreza extrema': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'más vulnerables': {'tipo': 'multiple_carencias_min', 'valor': 3},

            'personas con múltiples carencias': {'tipo': 'multiple_carencias_min', 'valor': 2},
            'población con múltiples carencias': {'tipo': 'multiple_carencias_min', 'valor': 2},
            'con múltiples carencias': {'tipo': 'multiple_carencias_min', 'valor': 2},
            'alta intensidad de carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            '3 o más carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'tienen 3 o más carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},

            # ==================== VARIABLES DEMOGRÁFICAS ====================
            'edad': {'tipo': 'columna', 'valor': 'edad_persona'},
            'años': {'tipo': 'columna', 'valor': 'edad_persona'},
            'parentesco': {'tipo': 'columna', 'valor': 'parentesco_persona'},
            'tipo de persona': {'tipo': 'columna', 'valor': 'tipo_persona'},
            'apoyos sociales': {'tipo': 'columna', 'valor': 'recibe_apoyos_sociales'},
            'recibe apoyos': {'tipo': 'columna', 'valor': 'recibe_apoyos_sociales'},
            'beneficiario': {'tipo': 'columna', 'valor': 'recibe_apoyos_sociales'},

           # ==================== PROGRAMAS SOCIALES (NOMBRES EXACTOS) ====================
            'pensión adultos mayores': {'tipo': 'programa', 'valor': 'pension_adultos_mayores'},
            'pensión mujeres': {'tipo': 'programa', 'valor': 'pension_mujeres_bienestar'},
            'mujeres bienestar': {'tipo': 'programa', 'valor': 'pension_mujeres_bienestar'},
            'becas primaria': {'tipo': 'programa', 'valor': 'beca_rita_cetina'},
            'beca rita cetina': {'tipo': 'programa', 'valor': 'beca_rita_cetina'},
            'rita cetina': {'tipo': 'programa', 'valor': 'beca_rita_cetina'},
            'benito juárez': {'tipo': 'programa', 'valor': 'beca_benito_juarez'},
            'beca secundaria': {'tipo': 'programa', 'valor': 'beca_benito_juarez'},
            'beca benito juárez': {'tipo': 'programa', 'valor': 'beca_benito_juarez'},
            'jóvenes escribiendo futuro': {'tipo': 'programa', 'valor': 'jovenes_escribiendo_el_futuro'},
            'jóvenes construyendo futuro': {'tipo': 'programa', 'valor': 'jovenes_construyendo_futuro'},
            'desde la cuna': {'tipo': 'programa', 'valor': 'desde_la_cuna'},
            'mi beca para empezar': {'tipo': 'programa', 'valor': 'mi_beca_para_empezar'},
            'imss bienestar': {'tipo': 'programa', 'valor': 'imss_bienestar'},
            'inea': {'tipo': 'programa', 'valor': 'inea'},
            'leche bienestar': {'tipo': 'programa', 'valor': 'leche_bienestar'},
            'seguro desempleo': {'tipo': 'programa', 'valor': 'seguro_desempleo_cdmx'},
            'ingreso ciudadano': {'tipo': 'programa', 'valor': 'ingreso_ciudadano_universal'},
            
            # ==================== GEOGRÁFICAS ====================
            'colonia': {'tipo': 'columna', 'valor': 'colonia'},
            'colonias': {'tipo': 'columna', 'valor': 'colonia'},
            'ageb': {'tipo': 'columna', 'valor': 'ageb'},
            'agebs': {'tipo': 'columna', 'valor': 'ageb'},
            'manzana': {'tipo': 'columna', 'valor': 'manzana'},
            'manzanas': {'tipo': 'columna', 'valor': 'manzana'},
            'ubicación': {'tipo': 'columna', 'valor': 'ubicacion'},
            'zona': {'tipo': 'columna', 'valor': 'ubicacion'},
            'localidad': {'tipo': 'columna', 'valor': 'ubicacion'},

             # ==================== TÉRMINOS DE ELEGIBILIDAD ====================
             'elegibles': {'tipo': 'concepto_elegibilidad', 'valor': 'elegibilidad'},
             'elegible': {'tipo': 'concepto_elegibilidad', 'valor': 'elegibilidad'},
             'pueden recibir': {'tipo': 'concepto_elegibilidad', 'valor': 'elegibilidad'},
             'califican para': {'tipo': 'concepto_elegibilidad', 'valor': 'elegibilidad'},
             'personas que pueden': {'tipo': 'concepto_elegibilidad', 'valor': 'elegibilidad'},

            # ==================== NUEVOS: TÉRMINOS GENERALES ====================
            'total personas': {'tipo': 'general', 'accion': 'conteo_total'},
            'cuántas personas': {'tipo': 'general', 'accion': 'conteo_total'},
            'número de personas': {'tipo': 'general', 'accion': 'conteo_total'},
            'cuántas hay': {'tipo': 'general', 'accion': 'conteo_total'},
            'total de personas': {'tipo': 'general', 'accion': 'conteo_total'},

            'personas por edad': {'tipo': 'columna', 'valor': 'edad_persona'},
            'distribución por edad': {'tipo': 'columna', 'valor': 'edad_persona'},
            'por edad': {'tipo': 'columna', 'valor': 'edad_persona'},

            'personas por sexo': {'tipo': 'columna', 'valor': 'sexo_persona'},
            'distribución por sexo': {'tipo': 'columna', 'valor': 'sexo_persona'},
            'por sexo': {'tipo': 'columna', 'valor': 'sexo_persona'},

            'por edad y sexo': {'tipo': 'tabla_cruzada', 'filas': 'edad_persona', 'columnas': 'sexo_persona'},
            'edad y sexo': {'tipo': 'tabla_cruzada', 'filas': 'edad_persona', 'columnas': 'sexo_persona'},
            'distribución por edad y sexo': {'tipo': 'tabla_cruzada', 'filas': 'edad_persona', 'columnas': 'sexo_persona'},
            # === 3 O MÁS CARENCIAS - FUNCIONA CON "O MÁS" ===
            '3 o más carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'tienen 3 o más carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'con 3 o más carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'más de 3 carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            'al menos 3 carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            '3+ carencias': {'tipo': 'multiple_carencias_min', 'valor': 3},
            
            # ==================== PARENTESCO/ROLES ====================
            # Jefe/Jefa de hogar
            "jefa de hogar": {'tipo': 'cruce', 'parentesco': 'Jefa de hogar', 'sexo': 'Mujer'},
            "jefas de hogar": {'tipo': 'cruce', 'parentesco': 'Jefa de hogar', 'sexo': 'Mujer'},
            "jefe de hogar": {'tipo': 'cruce', 'parentesco': 'Jefe de hogar', 'sexo': 'Hombre'},
            "jefes de hogar": {'tipo': 'cruce', 'parentesco': 'Jefe de hogar', 'sexo': 'Hombre'},
            
            # Madre
            "madre": {'tipo': 'parentesco', 'valor': 'Madre'},
            "madres": {'tipo': 'parentesco', 'valor': 'Madre'},
            "madre soltera": {'tipo': 'cruce', 'parentesco': 'Madre', 'sexo': 'Mujer'},
            "madres solteras": {'tipo': 'cruce', 'parentesco': 'Madre', 'sexo': 'Mujer'},
            
            # Padre
            "padre": {'tipo': 'parentesco', 'valor': 'Padre'},
            "padres": {'tipo': 'parentesco', 'valor': 'Padre'},
            "padre soltero": {'tipo': 'cruce', 'parentesco': 'Padre', 'sexo': 'Hombre'},
            "padres solteros": {'tipo': 'cruce', 'parentesco': 'Padre', 'sexo': 'Hombre'},
            
            # Hijo/Hija
            "hijo": {'tipo': 'parentesco', 'valor': 'Hijo'},
            "hijos": {'tipo': 'parentesco', 'valor': 'Hijo'},
            "hija": {'tipo': 'parentesco', 'valor': 'Hija'},
            "hijas": {'tipo': 'parentesco', 'valor': 'Hija'},
            "menores de edad": {'tipo': 'cruce', 'rango_edad': [0, 17]},
            
            # Abuelo/Abuela
            "abuelo": {'tipo': 'parentesco', 'valor': 'Abuelo'},
            "abuelos": {'tipo': 'parentesco', 'valor': 'Abuelo'},
            "abuela": {'tipo': 'parentesco', 'valor': 'Abuela'},
            "abuelas": {'tipo': 'parentesco', 'valor': 'Abuela'},
            "abuelos cuidadores": {'tipo': 'cruce', 'edad_min': 60, 'parentesco': 'Abuelo'},
            
            # Hermano/Hermana
            "hermano": {'tipo': 'parentesco', 'valor': 'Hermano'},
            "hermanos": {'tipo': 'parentesco', 'valor': 'Hermano'},
            "hermana": {'tipo': 'parentesco', 'valor': 'Hermana'},
            "hermanas": {'tipo': 'parentesco', 'valor': 'Hermana'},
            
            # Nieto/Nieta
            "nieto": {'tipo': 'parentesco', 'valor': 'Nieto'},
            "nietos": {'tipo': 'parentesco', 'valor': 'Nieto'},
            "nieta": {'tipo': 'parentesco', 'valor': 'Nieta'},
            "nietas": {'tipo': 'parentesco', 'valor': 'Nieta'},
            
            # Otro pariente
            "otro pariente": {'tipo': 'parentesco', 'valor': 'Otro pariente'},
            "otros parientes": {'tipo': 'parentesco', 'valor': 'Otro pariente'},
            "pariente": {'tipo': 'parentesco', 'valor': 'Otro pariente'},
            "parientes": {'tipo': 'parentesco', 'valor': 'Otro pariente'},
            
            # Sin parentesco
            "sin parentesco": {'tipo': 'parentesco', 'valor': 'Sin parentesco'},
            "sin familia": {'tipo': 'parentesco', 'valor': 'Sin parentesco'},
            
            # Dependiente (de tu lista de valores)
            "dependiente": {'tipo': 'parentesco', 'valor': 'Dependiente'},
            "dependientes": {'tipo': 'parentesco', 'valor': 'Dependiente'},
            "dependencia": {'tipo': 'parentesco', 'valor': 'Dependiente'},
            
            # Esposo/Esposa
            "esposo": {'tipo': 'parentesco', 'valor': 'Esposo'},
            "esposa": {'tipo': 'parentesco', 'valor': 'Esposa'},
            "cónyuge": {'tipo': 'parentesco', 'valor': 'Esposa'},  # O Esposo, ambos
            "pareja": {'tipo': 'parentesco', 'valor': 'Esposa'},
            
            # Suegra/Suegro
            "suegra": {'tipo': 'parentesco', 'valor': 'Suegra'},
            "suegro": {'tipo': 'parentesco', 'valor': 'Suegro'},
            "suegros": {'tipo': 'parentesco', 'valor': 'Suegra'},
 
        }
        
        return mapeo_grupos

    def traducir_consulta_natural(self, consulta: str) -> Dict[str, Any]:
        """
        VERSIÓN DEFINITIVA 2025 - 100% ALINEADA CON TUS TABLAS REALES
        - Devuelve "yes" / "no" tal como están en tu base de datos
        - Soporta múltiples sexos y rangos
        - Detecta negaciones perfectamente
        - Nunca pierde criterios
        """
        print(f"\n[TRADUCCIÓN] Procesando: {consulta}")

        criterios = {}
        variables_detectadas = set()
        terminos_mapeados = {}

        def _tiene_negacion(texto: str, termino: str, radio: int = 35) -> bool:
            pos = texto.find(termino)
            if pos == -1:
                return False
            inicio = max(0, pos - radio)
            fin = min(len(texto), pos + len(termino) + radio)
            fragmento = texto[inicio:fin]
            negaciones = [
                'sin ', ' no ', 'ningun', 'ninguna', 'ningún', 'ningún',
                'nunca ', 'jamás ', 'tampoco ', 'ni ', 'exento', 'exenta',
                'que no tenga', 'que no tengan', 'sin tener'
            ]
            return any(neg in fragmento for neg in negaciones)

        try:
            texto_consulta = consulta.lower()
            mapeo_completo = self._mapear_grupos_poblacionales()
            terminos_ordenados = sorted(mapeo_completo.keys(), key=len, reverse=True)

            # 1. Recolectar todos los términos
            terminos_encontrados = []
            for termino in terminos_ordenados:
                if termino in texto_consulta:
                    mapeo = mapeo_completo[termino]
                    terminos_encontrados.append((termino, mapeo))
                    terminos_mapeados[termino] = mapeo

            print(f"   Términos detectados: {len(terminos_encontrados)}")

            # 2. Procesar acumulando
            for termino_natural, mapeo in terminos_encontrados:
                tipo = mapeo.get('tipo')
                negada = _tiene_negacion(texto_consulta, termino_natural)
                print(f"   ↳ '{termino_natural}' → {tipo} {'(NEGADO)' if negada else ''}")

                # === GENERAL & TABLA CRUZADA ===
                if tipo == 'general':
                    criterios['accion_general'] = mapeo['accion']
                    variables_detectadas.add('general')

                elif tipo == 'tabla_cruzada':
                    criterios['tabla_cruzada'] = {'filas': mapeo['filas'], 'columnas': mapeo['columnas']}
                    variables_detectadas.update([mapeo['filas'], mapeo['columnas']])

                # === RANGO DE EDAD (múltiples) ===
                elif tipo == 'rango_edad':
                    if 'rangos_edad' not in criterios:
                        criterios['rangos_edad'] = []
                    criterios['rangos_edad'].append(mapeo['valor'])
                    variables_detectadas.add('edad_persona')

                # === SEXO (múltiples) ===
                elif tipo == 'sexo':
                    if 'sexo' not in criterios:
                        criterios['sexo'] = []
                    valor = mapeo['valor'].capitalize()  # Hombre / Mujer
                    if valor not in criterios['sexo']:
                        criterios['sexo'].append(valor)
                    variables_detectadas.add('sexo_persona')

                # === PROGRAMAS SOCIALES (con negación) ===
                elif tipo == 'programa':
                    programa = mapeo['valor']
                    columna = f"es_elegible_{programa}"
                    criterios[columna] = "no" if negada else "yes"
                    variables_detectadas.add(columna)
                    print(f"      → {columna} = {'no' if negada else 'yes'}")

                # === CARENCIAS ESPECÍFICAS (salud, rezago, seguridad social) ===
                elif tipo == 'columna':
                    term_low = termino_natural.lower()

                    if 'salud' in term_low:
                        col = 'presencia_carencia_salud_persona'
                        criterios[col] = "no" if negada else "yes"
                        variables_detectadas.add(col)

                    if any(x in term_low for x in ['educaci', 'rezago']):
                        col = 'presencia_rezago_educativo_persona'
                        criterios[col] = "no" if negada else "yes"
                        variables_detectadas.add(col)

                    if any(x in term_low for x in ['seguridad', 'social']):
                        col = 'presencia_carencia_seguridad_social_persona'
                        criterios[col] = "no" if negada else "yes"
                        variables_detectadas.add(col)

                # === PARENTESCO ===
                elif tipo == 'parentesco':
                    parentesco_valor = mapeo['valor']
                    criterios['parentesco'] = parentesco_valor
                    variables_detectadas.add('parentesco_persona')
                    print(f"      → Parentesco: {parentesco_valor}")

                # === MÚLTIPLES CARENCIAS (conteo) ===
                elif tipo == 'multiple_carencias_min':
                    if negada:
                        # "con menos de 3 carencias" → máximo 2
                        criterios['conteo_carencias_max'] = mapeo['valor'] - 1
                    else:
                        actual = criterios.get('conteo_carencias_min', 0)
                        criterios['conteo_carencias_min'] = max(actual, mapeo['valor'])
                    variables_detectadas.add('conteo_carencias_persona')

            # === FALLBACK: "X o más carencias" ===
            if 'conteo_carencias_min' not in criterios and 'conteo_carencias_max' not in criterios:
                if 'o más' in texto_consulta and 'carencias' in texto_consulta:
                    import re
                    match = re.search(r'(\d+)\s*o\s*m[áaá]+s?\s*carencias', texto_consulta, re.IGNORECASE)
                    if match:
                        n = int(match.group(1))
                        criterios['conteo_carencias_min'] = n
                        variables_detectadas.add('conteo_carencias_persona')

            # === GEOGRÁFICO ===
            for geo in ['ageb', 'colonia', 'ubicacion']:
                if f'por {geo}' in texto_consulta or f'por la {geo}' in texto_consulta:
                    criterios['segmentacion_geografica'] = geo
                    break

            # === ORDENAMIENTO ===
            if any(x in texto_consulta for x in ['mayor', 'más', 'top', 'principal']):
                criterios['ordenamiento'] = 'descendente'
            elif any(x in texto_consulta for x in ['menor', 'menos']):
                criterios['ordenamiento'] = 'ascendente'

            # === COMPATIBILIDAD ATRÁS ===
            if 'rangos_edad' in criterios and len(criterios['rangos_edad']) == 1:
                criterios['rango_edad'] = criterios['rangos_edad'][0]
            if 'sexo' in criterios and len(criterios['sexo']) == 1:
                criterios['sexo'] = criterios['sexo'][0]

            estado = 'éxito' if criterios or variables_detectadas else 'sin_criterios_detectados'

            resultado = {
                "consulta_original": consulta,
                "criterios_demograficos": criterios,
                "variables_detectadas": list(variables_detectadas),
                "terminos_mapeados": terminos_mapeados,
                "estado": estado
            }

            print(f"   FINAL → {len(criterios)} criterios, {len(variables_detectadas)} variables\n")
            return resultado

        except Exception as e:
            print(f"ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            return {
                "consulta_original": consulta,
                "criterios_demograficos": {},
                "variables_detectadas": [],
                "terminos_mapeados": {},
                "estado": "error",
                "error": str(e)
            }

    def validar_variables_mejorado(self, traduccion: Dict) -> Dict:
        """
        VALIDA que las variables detectadas existan en el dataset
        y que los criterios sean coherentes.
        PERMITE GENERALES Y TABLAS CRUZADAS SIN BLOQUEAR.
        """
        print(f"Validando traducción: {traduccion.get('terminos_mapeados', {})}")
        
        variables_detectadas = traduccion.get('variables_detectadas', [])
        criterios_demograficos = traduccion.get('criterios_demograficos', {})
        
        variables_validas = []
        variables_invalidas = []
        criterios_validos = {}
        
        # === COLUMNAS REALES DEL DATASET ===
        columnas_reales = set(self.df.columns)
        print(f"Columnas reales en dataset: {len(columnas_reales)} columnas")
        
        # === VALIDAR VARIABLES DETECTADAS ===
        for var in variables_detectadas:
            if var == 'general':
                variables_validas.append(var)
                print(f"Variable 'general' permitida (conteo total)")
            elif var in columnas_reales:
                variables_validas.append(var)
                print(f"Variable válida: {var}")
            else:
                variables_invalidas.append(var)
                print(f"Variable NO encontrada: {var}")
        
        # === VALIDAR CRITERIOS DEMOGRÁFICOS ===
        for criterio, valor in criterios_demograficos.items():
            # --- GENERALES Y TABLAS CRUZADAS ---
            if criterio == 'accion_general':
                criterios_validos[criterio] = valor
                print(f"Criterio general válido: {valor}")

            elif criterio == 'tabla_cruzada':
                filas = valor.get('filas')
                columnas = valor.get('columnas')
                if filas in columnas_reales and columnas in columnas_reales:
                    criterios_validos[criterio] = valor
                    variables_validas.extend([filas, columnas])  # Añade ambas columnas
                    print(f"Tabla cruzada válida: {filas} vs {columnas}")
                else:
                    print(f"Tabla cruzada inválida: {filas} o {columnas} no existen")

            # --- RANGO DE EDAD ---
            elif criterio == 'rango_edad':
                if isinstance(valor, list) and len(valor) == 2 and all(isinstance(x, int) for x in valor):
                    criterios_validos[criterio] = valor
                    variables_validas.append('edad_persona')  # ← COLUMNA REAL
                    print(f"Rango edad válido: {valor}")
                else:
                    print(f"Rango edad inválido: {valor}")

            # --- SEXO ---
            elif criterio == 'sexo':
                if valor in ['Hombre', 'Mujer']:
                    criterios_validos[criterio] = valor
                    variables_validas.append('sexo_persona')
                    print(f"Sexo válido: {valor}")

            # --- CARENCIAS INDIVIDUALES ---
            elif criterio in ['carencia_salud', 'carencia_educacion', 'carencia_seguridad_social']:
                criterios_validos[criterio] = valor
                variables_validas.append('conteo_carencias_persona')
                print(f"Carencia individual válida: {criterio}")

            # --- PROGRAMAS SOCIALES ---
            elif criterio == 'programa_social':
                col_programa = f"es_elegible_{valor}"
                if col_programa in columnas_reales:
                    criterios_validos[criterio] = valor
                    variables_validas.append(col_programa)
                    print(f"Programa válido: {valor} → {col_programa}")
                else:
                    print(f"Programa NO encontrado en dataset: {col_programa}")

            # --- SEGMENTACIÓN Y ORDENAMIENTO ---
            elif criterio in ['segmentacion_geografica', 'ordenamiento']:
                criterios_validos[criterio] = valor
                print(f"Criterio adicional válido: {criterio} = {valor}")

            # --- MÚLTIPLES CARENCIAS (EXACTAS O MÍNIMAS) ---
            elif criterio in ['multiple_carencias', 'multiple_carencias_min']:
                if isinstance(valor, int) and 1 <= valor <= 6:
                    criterios_validos[criterio] = valor
                    variables_validas.append('conteo_carencias_persona')  # ← CRUCIAL PARA EL LLM
                    print(f"Carencias múltiples válidas: {criterio} = {valor} carencias")
                else:
                    print(f"Valor inválido para carencias múltiples: {valor}")

            # --- CRITERIO DESCONOCIDO (debug) ---
            else:
                print(f"CRITERIO NO RECONOCIDO: {criterio} = {valor}")

        # === RESPUESTA PARA CONSULTAS GENERALES O TABLAS CRUZADAS ===
        if 'accion_general' in criterios_validos or 'tabla_cruzada' in criterios_validos:
            return {
                'ejecutable': True,
                'variables_validas': list(set(variables_validas + ['general'])),
                'variables_invalidas': variables_invalidas,
                'criterios_validos': criterios_validos,
                'estado': 'éxito_general'
            }

        # === RESPUESTA PARA CONSULTAS NORMALES (con variables válidas) ===
        if variables_validas:
            return {
                'ejecutable': True,
                'variables_validas': list(set(variables_validas)),
                'variables_invalidas': variables_invalidas,
                'criterios_validos': criterios_validos,
                'estado': 'éxito'
            }

        # === FALLBACK: NO EJECUTABLE ===
        return {
            'ejecutable': False,
            'variables_validas': variables_validas,
            'variables_invalidas': variables_invalidas,
            'criterios_validos': criterios_validos,
            'estado': 'fallo_validacion'
        }

    def analizar_flujo_completo(self, criterios_demograficos: Dict, 
                               segmentacion_geografica: str = None,
                               ordenamiento: str = "descendente", 
                               limite: int = 10) -> Dict:
        """
        FUNCIÓN PRINCIPAL CORREGIDA - Basada en estructura real de datos
        """
        try:
            print(f"🔍 Iniciando análisis con criterios: {criterios_demograficos}")
            
            # 1. DELIMITAR POBLACIÓN
            df_segmento = self.delimitador.aplicar_filtros(criterios_demograficos)
            total_segmento = len(df_segmento)
            
            resultados = {
                "flujo_analitico": "completo_corregido",
                "criterios_aplicados": criterios_demograficos,
                "metricas_principales": {
                    "total_personas_segmento": total_segmento,
                    "porcentaje_poblacion_total": round((total_segmento / len(self.df) * 100), 2) if len(self.df) > 0 else 0
                }
            }
            
            if total_segmento == 0:
                resultados["error"] = "No se encontraron personas con los criterios especificados"
                return resultados
            
            # 2. ANÁLISIS GEOGRÁFICO MEJORADO
            geo_col = segmentacion_geografica or criterios_demograficos.get('segmentacion_geografica')
            if geo_col and geo_col in df_segmento.columns:
                conteo_geografico = df_segmento[geo_col].value_counts()
                
                orden = criterios_demograficos.get('ordenamiento', ordenamiento)
                if orden == "descendente":
                    conteo_geografico = conteo_geografico.sort_values(ascending=False)
                else:
                    conteo_geografico = conteo_geografico.sort_values(ascending=True)
                
                top_geograficos = conteo_geografico.head(limite)
                
                resultados["analisis_geografico"] = {
                    "columna_geografica": geo_col,
                    "total_ubicaciones": len(conteo_geografico),
                    "top_ubicaciones": {
                        "nombres": top_geograficos.index.tolist(),
                        "conteos": top_geograficos.values.tolist(),
                        "porcentajes": (top_geograficos / total_segmento * 100).round(2).tolist()
                    }
                }
            
            # 3. PERFIL DEMOGRÁFICO
            if total_segmento > 0:
                resultados["perfil_demografico"] = self.demografico.generar_perfil_segmento(df_segmento)
            
            print(f"✅ Análisis completado exitosamente")
            return resultados
            
        except Exception as e:
            error_msg = f"Error en flujo analítico: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    # MÉTODOS DE COMPATIBILIDAD (se mantienen)
    def validar_variables(self, variables: List[str]) -> Dict[str, Any]:
        """Método legacy - Versión segura"""
        try:
            print("🔄 Usando validar_variables legacy seguro")
            
            # Crear estructura mínima para validar_variables_mejorado
            traduccion_simulada = {
                "criterios_demograficos": {"rango_edad_placeholder": True},  # Forzar criterios válidos
                "variables_detectadas": variables
            }
            
            # Usar el método mejorado
            resultado_mejorado = self.validar_variables_mejorado(traduccion_simulada)
            
            # Convertir a formato legacy
            return {
                "validas": resultado_mejorado["variables_validas"],
                "invalidas": resultado_mejorado["variables_invalidas"],
                "todas_validas": resultado_mejorado["ejecutable"]
            }
            
        except Exception as e:
            print(f"🔴 ERROR en validar_variables legacy: {str(e)}")
            return {
                "validas": [],
                "invalidas": variables,  # Marcar todas como inválidas por seguridad
                "todas_validas": False
            }
    
    def analizar_tabla_cruzada(self, variable_filas: str, variable_columnas: str, 
                            filtros: Dict = None, agrupar_edad: bool = True) -> Dict:
        """
        Genera tabla cruzada entre dos variables con filtros opcionales
        """
        print(f"🎯 [TABLA_CRUZADA] INICIANDO")
        print(f"   - Filas: {variable_filas}")
        print(f"   - Columnas: {variable_columnas}") 
        print(f"   - Filtros: {filtros}")
        print(f"   - Agrupar edad: {agrupar_edad}")
        
        try:
            # Validar variables
            if variable_filas not in self.df.columns:
                return {"error": f"Variable '{variable_filas}' no existe en el dataset"}
            if variable_columnas not in self.df.columns:
                return {"error": f"Variable '{variable_columnas}' no existe en el dataset"}
            
            # Aplicar filtros
            df_filtrado = self.df.copy()
            if filtros:
                df_filtrado = self._aplicar_filtros_tabla_cruzada(df_filtrado, filtros)
                print(f"✅ Filtros aplicados: {len(df_filtrado)} registros")
            
            if len(df_filtrado) == 0:
                return {"error": "No hay registros después de aplicar los filtros"}
            
            # Agrupar edades si es necesario
            if (variable_filas == 'edad_persona' or variable_columnas == 'edad_persona') and agrupar_edad:
                df_filtrado = self._agrupar_edades(df_filtrado)
                print(f"✅ Edades agrupadas en categorías")
            
            # Generar tabla cruzada
            tabla_cruzada = pd.crosstab(
                df_filtrado[variable_filas], 
                df_filtrado[variable_columnas],
                margins=True,
                margins_name="Total"
            )
            
            print(f"✅ Tabla generada: {tabla_cruzada.shape[0]-1} filas x {tabla_cruzada.shape[1]-1} columnas")
            
            # Convertir a formato JSON serializable
            tabla_dict = {}
            for fila in tabla_cruzada.index:
                tabla_dict[str(fila)] = {}
                for col in tabla_cruzada.columns:
                    valor = tabla_cruzada.loc[fila, col]
                    tabla_dict[str(fila)][str(col)] = int(valor) if pd.notna(valor) else 0
            
            # Generar representación en texto para el LLM
            tabla_texto = "TABLA CRUZADA:\n" + tabla_cruzada.to_string()
            
            # Calcular estadísticas adicionales
            total_general = int(tabla_cruzada.loc['Total', 'Total'])
            
            resultado = {
                "tipo_analisis": "tabla_cruzada",
                "variables": {
                    "filas": variable_filas,
                    "columnas": variable_columnas
                },
                "filtros_aplicados": filtros or {},
                "tabla_conteos": tabla_dict,
                "tabla_texto": tabla_texto,
                "resumen": {
                    "total_registros": total_general,
                    "total_categorias_filas": len(tabla_cruzada.index) - 1,
                    "total_categorias_columnas": len(tabla_cruzada.columns) - 1
                },
                "estado": "éxito"
            }
            
            print(f"🎉 [TABLA_CRUZADA] Completada exitosamente")
            return resultado
            
        except Exception as e:
            error_msg = f"Error generando tabla cruzada: {str(e)}"
            print(f"❌ [TABLA_CRUZADA] {error_msg}")
            import traceback
            print(traceback.format_exc())
            return {"error": error_msg}

    def analizar_distribucion_categorica(self, columna: str, top_n: int = 10) -> Dict[str, Any]:
        """Análisis de distribución para variables categóricas"""
        if columna not in self.df.columns:
            return {"error": f"Columna {columna} no encontrada"}
        
        datos = self.df[columna].dropna()
        conteo = datos.value_counts()
        porcentaje = (conteo / len(datos) * 100).round(2)
        
        return {
            "columna": columna, "tipo": "categorica", "total_registros": len(datos),
            "categorias_unicas": len(conteo), "distribucion": {
                "categorias": conteo.head(top_n).index.tolist(),
                "conteos": conteo.head(top_n).values.tolist(),
                "porcentajes": porcentaje.head(top_n).values.tolist()
            }
        }
    
    def analizar_distribucion_numerica(self, columna: str) -> Dict[str, Any]:
        """Análisis de distribución para variables numéricas"""
        if columna not in self.df.columns:
            return {"error": f"Columna {columna} no encontrada"}
        
        datos = self.df[columna].dropna()
        
        return {
            "columna": columna, "tipo": "numerica", "total_registros": len(datos),
            "estadisticas_descriptivas": {
                "media": float(datos.mean()), "mediana": float(datos.median()),
                "desviacion_estandar": float(datos.std()), "minimo": float(datos.min()),
                "maximo": float(datos.max()), "q1": float(datos.quantile(0.25)),
                "q3": float(datos.quantile(0.75))
            }
        }

    def explorar_ubicaciones_disponibles(self, top_n: int = 20) -> Dict[str, Any]:
        """Explora las ubicaciones geográficas disponibles en el dataset"""
        return {
            "colonias_mas_pobladas": {
                "total_colonias": self.df['colonia'].nunique(),
                "top_colonias": self.df['colonia'].value_counts().head(top_n).to_dict()
            },
            "agebs_mas_poblados": {
                "total_agebs": self.df['ageb'].nunique(), 
                "top_agebs": self.df['ageb'].value_counts().head(top_n).to_dict()
            },
            "ubicaciones_unicas": {
                "total_ubicaciones": self.df['ubicacion'].nunique(),
                "distribucion_ubicacion": self.df['ubicacion'].value_counts().to_dict()
            }
        }


    def _aplicar_filtros_tabla_cruzada(self, df: pd.DataFrame, filtros: Dict) -> pd.DataFrame:
        """Aplica filtros específicos para tablas cruzadas - MÉTODO FALTANTE"""
        df_filtrado = df.copy()
        
        # Mapeo de carencias a columnas reales
        carencia_map = {
            'carencia_salud': 'presencia_carencia_salud_persona',
            'carencia_educacion': 'presencia_rezago_educativo_persona', 
            'carencia_seguridad_social': 'presencia_carencia_seguridad_social_persona'
        }
        
        for filtro_key, filtro_value in filtros.items():
            if filtro_key in carencia_map and filtro_value:
                # Aplicar filtro de carencia
                columna_carencia = carencia_map[filtro_key]
                df_filtrado = df_filtrado[df_filtrado[columna_carencia] == 'yes']
                
            elif filtro_key == 'rango_edad' and filtro_value:
                # Aplicar filtro de rango de edad
                edad_min, edad_max = filtro_value
                df_filtrado = df_filtrado[
                    (df_filtrado['edad_persona'] >= edad_min) & 
                    (df_filtrado['edad_persona'] <= edad_max)
                ]
                
            elif filtro_key == 'sexo' and filtro_value:
                # Aplicar filtro de sexo
                df_filtrado = df_filtrado[df_filtrado['sexo_persona'] == filtro_value]
                
            elif filtro_key == 'programa_social' and filtro_value:
                # Aplicar filtro de programa social
                columna_programa = f"es_elegible_{filtro_value}"
                if columna_programa in df_filtrado.columns:
                    df_filtrado = df_filtrado[df_filtrado[columna_programa] == 'yes']
        
        return df_filtrado

    def _agrupar_edades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte edad numérica en grupos categóricos - MÉTODO FALTANTE"""
        df_modificado = df.copy()
        
        # Definir grupos de edad
        bins = [0, 18, 30, 45, 60, 75, 120]
        labels = ['0-17 años', '18-29 años', '30-44 años', '45-59 años', '60-74 años', '75+ años']
        
        # Crear columna temporal para edad agrupada
        df_modificado['edad_grupo'] = pd.cut(
            df_modificado['edad_persona'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        # Reemplazar la columna original de edad
        df_modificado = df_modificado.drop('edad_persona', axis=1)
        df_modificado = df_modificado.rename(columns={'edad_grupo': 'edad_persona'})
        
        return df_modificado

# ============================================================================
# 4. AGENTE LLM ACTUALIZADO (VERSIÓN LIMPIA Y EN ORDEN)
# ============================================================================
class AgenteAnaliticoLLM:
    """Agente que usa LLM + Function Calling con sistema de robustez mejorado - VERSIÓN ACTUALIZADA"""
    def __init__(self, df_completo, api_key: str):
        self.df = df_completo
        self.analizador = AnalizadorUnidimensional(df_completo, api_key)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60.0)
        
        # === PASO 1: CALCULAR CIFRAS REALES ===
        total_personas = len(df_completo)
        total_hogares = df_completo['id_hogar'].nunique()
        total_colonias = df_completo['colonia'].nunique()
        total_agebs = df_completo['ageb'].nunique()
        
        # === PASO 2: CONSTRUIR TU CONTEXTO COMPLETO ===
        contexto = f"""
        Eres un analista especializado en política social con capacidades avanzadas de análisis de brechas.
        
        DATASET DISPONIBLE (CIFRAS REALES):
        - {total_personas:,} personas en {total_hogares:,} hogares
        - Promedio {total_personas / total_hogares:.2f} personas por hogar
        - Carencias: salud, educación, seguridad social
        - 13 programas sociales de elegibilidad
        - Datos geográficos: {total_colonias} colonias, {total_agebs} AGEBs
        
        CAPACIDADES PRINCIPALES:
        1. ANÁLISIS DE BRECHAS: Identificar personas elegibles que NO reciben programas
        2. CARENCIAS SIN COBERTURA: Personas con carencias sin programas relacionados  
        3. INTENSIDAD DE CARENCIAS: Personas con múltiples carencias simultáneas
        4. COMPARATIVA PROGRAMAS: Analizar múltiples programas simultáneamente
        5. DELIMITACIÓN POBLACIONAL: Segmentar por edad, sexo, ubicación, carencias
        6. ANÁLISIS GEOGRÁFICO: Distribución por AGEB, colonia, ubicación
        
        INSTRUCCIONES:
        - PARA BRECHAS → Usa 'analizar_brechas_programa_grupo'
        - PARA CARENCIAS SIN COBERTURA → Usa 'identificar_carencias_sin_cobertura'  
        - PARA MÚLTIPLES CARENCIAS → Usa 'analizar_intensidad_carencias'
        - PARA COMPARAR PROGRAMAS → Usa 'analizar_brechas_multiprograma'
        - PARA ANÁLISIS GENERAL → Usa 'analizar_flujo_completo'
        - PARA CONSULTAS GEOGRÁFICAS → Usa 'segmentacion_geografica' en analizar_flujo_completo
        
        Responde de manera natural y enfocada en insights accionables.
        """.strip()

        # === PASO 3: INICIALIZAR messages CON TU CONTEXTO ===
        self.messages = [{"role": "system", "content": contexto}]

    def _safe_json_serialize(self, obj):
        """
        Convierte tipos numpy a tipos Python nativos.
        Evita error 400 cuando DeepSeek rechaza JSON con numpy types.
        
        CRÍTICA: Previene colapso de sesiones después de 8-10 consultas
        
        Uso:
            resultado_limpio = self._safe_json_serialize(resultado)
            json_str = json.dumps(resultado_limpio, default=str)
        
        Ejemplos:
            numpy.int64(100)     → int(100)
            numpy.float64(35.5)  → float(35.5)
            numpy.array([1,2,3]) → [1, 2, 3]
            pd.Series({...})     → dict
        """
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: self._safe_json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._safe_json_serialize(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    # ***** CORREGIDO Y ESTANDARIZADO *****
    def _definir_herramientas_analisis(self):
        """Define las funciones disponibles para el LLM - VERSIÓN CORREGIDA Y ESTANDARIZADA"""
        tools = [
            # HERRAMIENTA PRINCIPAL ACTUALIZADA
            {
                "type": "function",
                "function": {
                    "name": "analizar_flujo_completo",
                    "description": "ANÁLISIS GENERAL: Delimitación poblacional completa con análisis demográfico y geográfico. Úsala para consultas generales de segmentación, distribución geográfica y perfiles poblacionales.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "criterios_demograficos": {
                                "type": "object",
                                "description": "Criterios para delimitar población",
                                "properties": {
                                    "rango_edad": {"type": "array", "items": {"type": "number"}, "description": "Rango de edad [min, max]. Ej: [65, 100] para adultos mayores, [0, 12] para niños"},
                                    "sexo": {"type": "string", "enum": ["Mujer", "Hombre"], "description": "Sexo de la persona"},
                                    "carencia_salud": {"type": "boolean", "description": "Filtrar por carencia en salud (presencia_carencia_salud_persona = 'yes')"},
                                    "carencia_educacion": {"type": "boolean", "description": "Filtrar por carencia en educación (presencia_rezago_educativo_persona = 'yes')"},
                                    "carencia_seguridad_social": {"type": "boolean", "description": "Filtrar por carencia en seguridad social (presencia_carencia_seguridad_social_persona = 'yes')"},
                                    "programa_social": {"type": "string", "description": "Programa social específico: pension_adultos_mayores, pension_mujeres_bienestar, beca_benito_juarez, etc."},
                                    "ubicacion": {"type": "string", "description": "Ubicación para filtrar (colonia, ageb, zona)"},
                                    "segmentacion_geografica": {"type": "string", "description": "Columna para segmentación: ageb, colonia, ubicacion"},
                                    "ordenamiento": {"type": "string", "enum": ["ascendente", "descendente"], "description": "Ordenamiento de resultados"}
                                }
                            },
                            "segmentacion_geografica": {
                                "type": "string", 
                                "description": "Columna para segmentación geográfica: ageb, colonia, ubicacion. Úsala para consultas como 'por ageb', 'por colonia'",
                                "default": None
                            },
                            "ordenamiento": {
                                "type": "string",
                                "enum": ["ascendente", "descendente"],
                                "description": "Ordenamiento de resultados. 'descendente' para mayor a menor, 'ascendente' para menor a mayor",
                                "default": "descendente"
                            },
                            "limite": {
                                "type": "integer",
                                "description": "Límite de resultados a mostrar en rankings geográficos", 
                                "default": 10
                            }
                        },
                        "required": ["criterios_demograficos"]
                    }
                }
            },
            # ============================================================================
            # NUEVAS HERRAMIENTAS DE ELEGIBILIDAD ESPECÍFICA
            # ============================================================================
            {
                "type": "function",
                "function": {
                    "name": "analizar_elegibilidad_programa",
                    "description": "ANÁLISIS DIRECTO DE ELEGIBILIDAD: Analiza personas elegibles para programas sociales específicos. Úsala para: 'personas elegibles para X', 'elegibles por edad/ubicación/sexo', 'análisis por AGEB/colonia', 'personas que pueden recibir programa'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "programa": {
                                "type": "string", 
                                "description": "Nombre EXACTO del programa: imss_bienestar, pension_adultos_mayores, pension_mujeres_bienestar, beca_benito_juarez, beca_rita_cetina, jovenes_escribiendo_el_futuro, jovenes_construyendo_futuro, desde_la_cuna, mi_beca_para_empezar, seguro_desempleo_cdmx, ingreso_ciudadano_universal, inea, leche_bienestar",
                                "enum": ["imss_bienestar", "pension_adultos_mayores", "pension_mujeres_bienestar", 
                                    "beca_benito_juarez", "beca_rita_cetina", "jovenes_escribiendo_el_futuro",
                                    "jovenes_construyendo_futuro", "desde_la_cuna", "mi_beca_para_empezar",
                                    "seguro_desempleo_cdmx", "ingreso_ciudadano_universal", "inea", "leche_bienestar"]
                            },
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"},
                                "description": "Rango de edad [min, max]. Ej: [0, 12] para niños, [65, 100] para adultos mayores",
                                "default": None
                            },
                            "ubicacion": {
                                "type": "string",
                                "description": "Ubicación para filtrar: colonia, AGEB, zona específica",
                                "default": None
                            },
                            "sexo": {
                                "type": "string",
                                "enum": ["Mujer", "Hombre"],
                                "description": "Filtrar por sexo",
                                "default": None
                            },
                            "carencia": {
                                "type": "string", 
                                "enum": ["salud", "educacion", "seguridad_social"],
                                "description": "Filtrar por tipo de carencia",
                                "default": None
                            },
                            "segmentacion_geografica": {
                                "type": "string",
                                "enum": ["ageb", "colonia"],
                                "description": "Segmentar resultados por nivel geográfico. Úsala para 'por AGEB', 'por colonia'",
                                "default": None
                            },
                            "incluir_brecha": {
                                "type": "boolean",
                                "description": "Incluir análisis de brecha (personas elegibles que no reciben apoyo)",
                                "default": True
                            }
                        },
                        "required": ["programa"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analizar_cobertura_geografica",
                    "description": "ANÁLISIS GEOGRÁFICO ESPECÍFICO: Analiza distribución de elegibles por AGEB o colonia. Úsala para: 'elegibles por AGEB', 'cobertura geográfica por colonia', 'densidad de elegibles por zona', 'distribución territorial de programa'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "programa": {
                                "type": "string",
                                "description": "Nombre del programa social",
                                "enum": ["imss_bienestar", "pension_adultos_mayores", "pension_mujeres_bienestar", 
                                    "beca_benito_juarez", "beca_rita_cetina", "jovenes_escribiendo_el_futuro",
                                    "jovenes_construyendo_futuro", "desde_la_cuna", "mi_beca_para_empezar",
                                    "seguro_desempleo_cdmx", "ingreso_ciudadano_universal", "inea", "leche_bienestar"]
                            },
                            "nivel_geografico": {
                                "type": "string",
                                "enum": ["ageb", "colonia"],
                                "description": "Nivel geográfico para el análisis. 'ageb' para análisis por AGEB, 'colonia' por colonia",
                                "default": "ageb"
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Número de zonas top a mostrar",
                                "default": 10
                            }
                        },
                        "required": ["programa"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analizar_elegibilidad_multiple",
                    "description": "COMPARATIVA DE ELEGIBILIDAD: Analiza y compara múltiples programas simultáneamente. Úsala para: 'comparar elegibilidad entre programas', 'qué programa tiene más elegibles', 'ranking de programas por cobertura', 'comparar pensiones vs becas'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "programas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista de programas a comparar. Ej: [pension_adultos_mayores, pension_mujeres_bienestar], [beca_benito_juarez, beca_rita_cetina]",
                                "enum": ["imss_bienestar", "pension_adultos_mayores", "pension_mujeres_bienestar", 
                                    "beca_benito_juarez", "beca_rita_cetina", "jovenes_escribiendo_el_futuro",
                                    "jovenes_construyendo_futuro", "desde_la_cuna", "mi_beca_para_empezar",
                                    "seguro_desempleo_cdmx", "ingreso_ciudadano_universal", "inea", "leche_bienestar"]
                            },
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"},
                                "description": "Rango de edad opcional [min, max]",
                                "default": None
                            },
                            "ubicacion": {
                                "type": "string",
                                "description": "Ubicación para filtrar",
                                "default": None
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Número máximo de programas en ranking",
                                "default": 5
                            }
                        },
                        "required": ["programas"]
                    }
                }
            },
            # ============================================================================
            # HERRAMIENTAS EXISTENTES DE BRECHAS (MANTENIDAS)
            # ============================================================================
            {
                "type": "function",
                "function": {
                    "name": "analizar_brechas_programa_grupo",
                    "description": "ANÁLISIS DE BRECHAS: Identifica personas elegibles que NO reciben un programa específico. Úsala para: 'adultos mayores sin pensión', 'personas elegibles que no reciben X programa', 'brechas de cobertura por edad/ubicación'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "programa": {
                                "type": "string", 
                                "description": "Nombre del programa social",
                                "enum": ["imss_bienestar", "pension_adultos_mayores", "pension_mujeres_bienestar", 
                                    "beca_benito_juarez", "beca_rita_cetina", "jovenes_escribiendo_el_futuro",
                                    "jovenes_construyendo_futuro", "desde_la_cuna", "mi_beca_para_empezar",
                                    "seguro_desempleo_cdmx", "ingreso_ciudadano_universal", "inea", "leche_bienestar"]
                            },
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"}, 
                                "description": "Rango de edad [min, max]. Ej: [65, 100] para adultos mayores, [0, 12] para niños", 
                                "default": None
                            },
                            "ubicacion": {
                                "type": "string", 
                                "description": "Ubicación para filtrar: colonia, ageb, zona", 
                                "default": None
                            }
                        },
                        "required": ["programa"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "identificar_carencias_sin_cobertura",
                    "description": "CARENCIAS SIN COBERTURA: Personas con carencias que NO son elegibles para programas relacionados. Úsala para: 'carencia de salud sin programas', 'personas con rezago educativo sin becas', 'vulnerabilidad sin protección social'",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "carencia": {
                                "type": "string", 
                                "enum": ["salud", "educacion", "seguridad_social"], 
                                "description": "Tipo de carencia a analizar"
                            },
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"}, 
                                "description": "Rango de edad opcional [min, max]", 
                                "default": None
                            }
                        },
                        "required": ["carencia"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analizar_intensidad_carencias", 
                    "description": "INTENSIDAD DE CARENCIAS: Analiza personas con múltiples carencias simultáneas. Úsala para: 'niños con mayor carencia social', 'personas con vulnerabilidad extrema', 'múltiples carencias por edad'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"}, 
                                "description": "Rango de edad [min, max]. Ej: [0, 12] para niños, [65, 100] para adultos mayores", 
                                "default": None
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analizar_brechas_multiprograma", 
                    "description": "COMPARATIVA DE BRECHAS: Analiza y compara brechas entre múltiples programas. Úsala para: 'comparar pensiones adultos mayores vs mujeres', 'qué programa tiene mayor brecha', 'análisis comparativo de cobertura'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "programas": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "Lista de programas a comparar",
                                "enum": ["imss_bienestar", "pension_adultos_mayores", "pension_mujeres_bienestar", 
                                    "beca_benito_juarez", "beca_rita_cetina", "jovenes_escribiendo_el_futuro",
                                    "jovenes_construyendo_futuro", "desde_la_cuna", "mi_beca_para_empezar",
                                    "seguro_desempleo_cdmx", "ingreso_ciudadano_universal", "inea", "leche_bienestar"]
                            },
                            "rango_edad": {
                                "type": "array", 
                                "items": {"type": "number"}, 
                                "description": "Rango de edad opcional [min, max]", 
                                "default": None
                            }
                        },
                        "required": ["programas"]
                    }
                }
            },
            # ============================================================================
            # HERRAMIENTAS COMPLEMENTARIAS (MANTENIDAS)
            # ============================================================================
            {
                "type": "function",
                "function": {
                    "name": "analizar_distribucion_categorica",
                    "description": "Análisis simple de distribución de variables categóricas. Úsala para: 'distribución por sexo', 'tipos de parentesco', 'ubicaciones disponibles'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columna": {
                                "type": "string", 
                                "description": "Nombre de la columna categórica: sexo_persona, parentesco_persona, colonia, ageb, etc."
                            }
                        },
                        "required": ["columna"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "analizar_distribucion_numerica",
                    "description": "Análisis simple de distribución de variables numéricas. Úsala para: 'estadísticas de edad', 'distribución de personas por hogar'",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "columna": {
                                "type": "string", 
                                "description": "Nombre de la columna numérica: edad_persona, personas, total_personas, etc."
                            }
                        },
                        "required": ["columna"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explorar_ubicaciones_disponibles",
                    "description": "Explorar colonias y AGEBs disponibles en el dataset. Úsala para conocer las ubicaciones geográficas del dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer", 
                                "description": "Número máximo de ubicaciones a mostrar", 
                                "default": 20
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analizar_tabla_cruzada",
                    "description": "TABLAS CRUZADAS: Genera análisis de distribución conjunta entre dos variables. Úsala para: 'por edad y sexo', 'tabla cruzada entre X e Y', 'distribución conjunta', 'clasificación múltiple', 'cross tabulation'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "variable_filas": {
                                "type": "string", 
                                "description": "Variable para las filas: edad_persona, sexo_persona, parentesco_persona, colonia, ageb, etc."
                            },
                            "variable_columnas": {
                                "type": "string",
                                "description": "Variable para las columnas: sexo_persona, presencia_carencia_salud_persona, recibe_apoyos_sociales, etc."
                            },
                            "filtros": {
                                "type": "object",
                                "description": "Filtros opcionales para segmentar la población",
                                "properties": {
                                    "carencia_salud": {"type": "boolean"},
                                    "carencia_educacion": {"type": "boolean"},
                                    "carencia_seguridad_social": {"type": "boolean"},
                                    "rango_edad": {"type": "array", "items": {"type": "number"}},
                                    "sexo": {"type": "string", "enum": ["Mujer", "Hombre"]},
                                    "programa_social": {"type": "string"}
                                }
                            },
                            "agrupar_edad": {
                                "type": "boolean", 
                                "description": "Convertir edad numérica en grupos categóricos",
                                "default": True
                            }
                        },
                        "required": ["variable_filas", "variable_columnas"]
                    }
                }
            }
            

        ]
        return tools

    # ============================================================================
    # MÉTODO PRINCIPAL ACTUALIZADO
    # ============================================================================
    def procesar_consulta(self, consulta_usuario: str) -> str:
        """Procesa consulta en lenguaje natural - VERSIÓN CON DEBUG COMPLETO"""
        self.messages.append({"role": "user", "content": consulta_usuario})
        
        try:
            # Primera llamada - LLM decide qué función usar
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=self._definir_herramientas_analisis(),
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            self.messages.append(response_message)
            
            if response_message.tool_calls:
                print("🤖 LLM solicitó usar función de análisis...")
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Ejecutando {function_name} con args: {function_args}")
                    
                    # ============================================================
                    # CRÍTICO: Agregar manejo de analizar_tabla_cruzada
                    # ============================================================
                    if function_name == "analizar_tabla_cruzada":
                        print(f"📊 [TABLA_CRUZADA] Procesando solicitud...")
                        result = self.analizador.analizar_tabla_cruzada(**function_args)
                        
                        # DEBUG: Verificar qué devolvió la función
                        print(f"🔍 [DEBUG] Tipo de resultado: {type(result)}")
                        print(f"🔍 [DEBUG] Es dict: {isinstance(result, dict)}")
                        
                        if isinstance(result, dict):
                            print(f"🔍 [DEBUG] Keys del resultado: {list(result.keys())}")
                            print(f"🔍 [DEBUG] Estado: {result.get('estado', 'NO DEFINIDO')}")
                            
                            if 'error' in result:
                                print(f"❌ [DEBUG] Error encontrado: {result['error']}")
                            
                            if 'tabla_texto' in result:
                                print(f"✅ [DEBUG] Tabla texto presente (longitud: {len(result['tabla_texto'])} chars)")
                                print(f"📋 [DEBUG] Primeros 200 chars de tabla:\n{result['tabla_texto'][:200]}")
                            
                            if 'resumen' in result:
                                print(f"📊 [DEBUG] Resumen: {result['resumen']}")
                        else:
                            print(f"⚠️ [DEBUG] Resultado NO es un diccionario!")
                        
                        print(f"✅ [TABLA_CRUZADA] Resultado obtenido")
                        
                    elif function_name == "analizar_flujo_completo":
                        result = self.analizador.analizar_flujo_completo(**function_args)
                    
                    # NUEVAS HERRAMIENTAS DE ELEGIBILIDAD
                    elif function_name == "analizar_elegibilidad_programa":
                        result = self.analizador.programas.analizar_elegibilidad_programa(**function_args)
                    elif function_name == "analizar_cobertura_geografica":
                        result = self.analizador.programas.analizar_cobertura_geografica(**function_args)
                    elif function_name == "analizar_elegibilidad_multiple":
                        result = self.analizador.programas.analizar_elegibilidad_multiple(**function_args)
                    
                    # HERRAMIENTAS EXISTENTES DE BRECHAS
                    elif function_name == "analizar_brechas_programa_grupo":
                        result = self.analizador.programas.analizar_brechas_programa_grupo(**function_args)
                    elif function_name == "identificar_carencias_sin_cobertura":
                        result = self.analizador.programas.identificar_carencias_sin_cobertura(**function_args)
                    elif function_name == "analizar_intensidad_carencias":
                        result = self.analizador.programas.analizar_intensidad_carencias(**function_args)
                    elif function_name == "analizar_brechas_multiprograma":
                        result = self.analizador.programas.analizar_brechas_multiprograma(**function_args)
                    
                    # HERRAMIENTAS COMPLEMENTARIAS
                    elif function_name == "analizar_distribucion_categorica":
                        result = self.analizador.analizar_distribucion_categorica(**function_args)
                    elif function_name == "analizar_distribucion_numerica":
                        result = self.analizador.analizar_distribucion_numerica(**function_args)
                    elif function_name == "explorar_ubicaciones_disponibles":
                        result = self.analizador.explorar_ubicaciones_disponibles(**function_args)
                    else:
                        result = {"error": f"Función {function_name} no reconocida"}
                    
                    # DEBUG: Verificar antes de agregar al contexto
                    print(f"🔍 [DEBUG] Agregando resultado al contexto del LLM...")
                    print(f"🔍 [DEBUG] Tool call ID: {tool_call.id}")
                    print(f"🔍 [DEBUG] Function name: {function_name}")
                    
                    resultado_limpio=self._safe_json_serialize(result)
                    # Agregar resultado al contexto
                    self.messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(resultado_limpio, ensure_ascii=False,default=str)
                    })
                    
                    print(f"✅ [JSON SEGURO] {function_name} - Tipos convertidos")
                
                # DEBUG: Verificar mensajes antes de segunda llamada
                print(f"🔍 [DEBUG] Total de mensajes en contexto: {len(self.messages)}")
                print(f"🔍 [DEBUG] Último mensaje es de tipo: {self.messages[-1].get('role', 'UNKNOWN')}")
                
                # Respuesta final con contexto
                print(f"🤖 [DEBUG] Solicitando respuesta final al LLM...")
                second_response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=self.messages
                )
                
                final_response = second_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": final_response})
                
                print(f"✅ [DEBUG] Respuesta final generada")
                return final_response
            else:
                return response_message.content
                
        except Exception as e:
            error_msg = f"❌ Error en el análisis: {str(e)}"
            print(error_msg)
            import traceback
            print(f"🔴 Traceback: {traceback.format_exc()}")
            return error_msg

    # ============================================================================
    # SISTEMA DE ROBUSTEZ MEJORADO
    # ============================================================================

    def detectar_ambiguedades(self, consulta: str) -> Dict[str, Any]:
        """Identifica términos ambiguos en la consulta y sugiere clarificaciones"""
        
        patrones_ambiguos = {
            'pobreza': {
                'termino': 'pobreza',
                'opciones': ['pobreza por ingresos', 'pobreza por carencias', 'pobreza multidimensional'],
                'pregunta': "¿Qué tipo de pobreza le interesa analizar?"
            },
            'vulnerable': {
                'termino': 'vulnerable', 
                'opciones': ['vulnerable por edad', 'vulnerable por carencias', 'vulnerable por discapacidad'],
                'pregunta': "¿Vulnerabilidad por qué característica?"
            },
            'prioritario': {
                'termino': 'prioritario',
                'opciones': ['prioritario para salud', 'prioritario para educación', 'prioritario para pensiones'],
                'pregunta': "¿Prioritario para qué área o programa?"
            },
            'cobertura': {
                'termino': 'cobertura',
                'opciones': ['cobertura de salud', 'cobertura educativa', 'cobertura de seguridad social'],
                'pregunta': "¿Cobertura de qué servicio le interesa?"
            },
            'beneficiario': {
                'termino': 'beneficiario',
                'opciones': ['beneficiario de pensiones', 'beneficiario de becas', 'beneficiario de salud'],
                'pregunta': "¿Beneficiario de qué tipo de programa?"
            },
            'acceso': {
                'termino': 'acceso',
                'opciones': ['acceso a salud', 'acceso a educación', 'acceso a seguridad social'],
                'pregunta': "¿Acceso a qué servicio le interesa?"
            }
        }
        
        ambiguedades_detectadas = []
        
        for clave, patron in patrones_ambiguos.items():
            if patron['termino'] in consulta.lower():
                ambiguedades_detectadas.append(patron)
        
        return {
            'hay_ambiguedad': len(ambiguedades_detectadas) > 0,
            'ambiguedades': ambiguedades_detectadas,
            'consulta_original': consulta
        }

    def generar_respuesta_clarificacion(self, ambiguedades: List[Dict]) -> str:
        """Genera una respuesta amigable pidiendo clarificación"""
        
        if not ambiguedades:
            return None
        
        respuesta = "🤔 **Necesito clarificar tu consulta:**\n\n"
        
        for i, ambiguedad in enumerate(ambiguedades):
            respuesta += f"**{ambiguedad['pregunta']}**\n"
            
            for opcion in ambiguedad['opciones']:
                respuesta += f"• {opcion}\n"
            
            if i < len(ambiguedades) - 1:
                respuesta += "\n"
        
        respuesta += "\n💡 **Puedes reformular tu pregunta como:**\n"
        respuesta += "• 'Personas vulnerables por edad sin pensiones'\n" 
        respuesta += "• 'Brechas de cobertura en servicios de salud'\n"
        respuesta += "• 'Población prioritaria para programas educativos'\n"
        respuesta += "• 'Beneficiarios de pensiones por grupo de edad'"
        
        return respuesta

    def _generar_sugerencias_contextuales_mejoradas(self, consulta: str) -> str:
        """Sugerencias más inteligentes basadas en el contexto de la consulta"""
        
        sugerencias_especificas = []
        sugerencias_generales = [
            "• 'Adultos mayores sin pensión'",
            "• 'Niños con múltiples carencias'", 
            "• 'Personas con carencia de salud sin programas'",
            "• 'Mujeres elegibles para pensión que no la reciben'",
            "• 'Comparar brechas entre programas sociales'",
            "• 'Distribución de carencias por edad y sexo'"
        ]
        
        # Detectar contexto para sugerencias específicas
        consulta_lower = consulta.lower()
        
        if any(palabra in consulta_lower for palabra in ['salud', 'médico', 'hospital', 'enfermedad', 'acceso salud']):
            sugerencias_especificas = [
                "• 'Personas con carencia de acceso a salud'",
                "• 'Cobertura de servicios médicos por edad'", 
                "• 'Brechas en programas de salud por colonia'",
                "• 'Distribución de carencia de salud por sexo'"
            ]
        
        elif any(palabra in consulta_lower for palabra in ['educación', 'educacion', 'escuela', 'estudio', 'beca']):
            sugerencias_especificas = [
                "• 'Niños con rezago educativo'",
                "• 'Distribución de becas por nivel educativo'", 
                "• 'Jóvenes sin acceso a educación superior'",
                "• 'Rezago educativo por grupo de edad y sexo'"
            ]
        
        elif any(palabra in consulta_lower for palabra in ['pensión', 'pension', 'jubilación', 'adulto mayor', 'tercera edad']):
            sugerencias_especificas = [
                "• 'Adultos mayores sin pensión'",
                "• 'Brechas en cobertura de pensiones'",
                "• 'Mujeres mayores sin seguridad social'",
                "• 'Distribución de pensiones por colonia'"
            ]
        
        elif any(palabra in consulta_lower for palabra in ['carencia', 'vulnerabilidad', 'necesidad']):
            sugerencias_especificas = [
                "• 'Personas con múltiples carencias'",
                "• 'Intensidad de carencias por edad'",
                "• 'Carencias sin cobertura de programas'",
                "• 'Distribución de carencias por zona'"
            ]
        
        # Combinar sugerencias (máximo 6)
        todas_sugerencias = (sugerencias_especificas + sugerencias_generales)[:6]
        
        return f"""🤔 No identifiqué variables específicas en: "{consulta}"

💡 **Basado en tu consulta, te sugiero:**

{"".join(todas_sugerencias)}

🎯 **Puedes ser más específico como:**
• "¿Cuántos adultos mayores de 65 años no reciben pensión?"
• "Personas con carencia de salud por sexo y edad"  
• "Comparar brechas entre becas Benito Juárez y Rita Cetina"

¿Sobre qué aspecto te gustaría analizar?"""

    def _generar_respuesta_variables_invalidas(self, validacion: Dict, consulta: str) -> str:
        """Maneja cuando se detectan variables inválidas"""
        
        return f"""❌ **No pude encontrar algunas variables en la base de datos**

**Consulta:** {consulta}

**Variables inválidas detectadas:** {', '.join(validacion['invalidas'])}

**Variables válidas disponibles:** {', '.join(validacion['validas'])}

💡 **Puedes usar términos como:**
• 'salud', 'educación', 'seguridad social' para carencias
• 'adultos mayores', 'niños', 'mujeres' para grupos de edad  
• 'pensión adultos mayores', 'becas', 'programas mujeres' para programas
• 'colonia', 'zona', 'ubicación' para análisis geográfico

¿Podrías reformular tu consulta usando estos términos?"""

    def _generar_respuesta_error_amigable(self, error: Exception, consulta: str) -> str:
        """Genera respuestas útiles cuando ocurren errores"""
        
        mensajes_error = {
            "JSONDecodeError": "❌ **Error de datos:** Hubo un problema procesando la información.",
            "KeyError": "🔧 **Error técnico:** No encontré algunas variables necesarias.",
            "ConnectionError": "🌐 **Error de conexión:** No puedo conectar con el servicio de análisis.",
            "TimeoutError": "⏰ **Tiempo de espera:** La consulta está tomando más tiempo de lo esperado.",
            "AttributeError": "⚙️ **Error del sistema:** Hay un problema temporal con mis funciones."
        }
        
        nombre_error = type(error).__name__
        mensaje_especifico = mensajes_error.get(nombre_error, "❌ **Ocurrió un error inesperado**")
        
        respuesta = f"{mensaje_especifico}\n\n"
        respuesta += f"**Consulta:** {consulta}\n\n"
        respuesta += "🔄 **Puedes intentar:**\n"
        respuesta += "• Reformular tu pregunta de otra manera\n"
        respuesta += "• Usar términos más específicos\n" 
        respuesta += "• Esperar unos segundos y volver a intentar\n\n"
        respuesta += "💡 **Ejemplo seguro:** 'Adultos mayores sin pensión por colonia'"
        
        return respuesta

    def procesar_consulta_mejorado(self, consulta_usuario: str) -> str:
        """
        PROCEDE CON CONSULTA:
        - Detecta ambigüedades
        - Traduce
        - Valida (solo si no es general)
        - Pasa al LLM si es válida o general
        """
        print(f"\nUSUARIO: {consulta_usuario}")
        
        try:
            # === 1. DETECTAR AMBIGÜEDADES ===
            analisis_ambiguedad = self.detectar_ambiguedades(consulta_usuario)
            if analisis_ambiguedad['hay_ambiguedad']:
                return self.generar_respuesta_clarificacion(analisis_ambiguedad['ambiguedades'])

            # === 2. TRADUCCIÓN ===
            traduccion = self.analizador.traducir_consulta_natural(consulta_usuario)
            print(f"Auto-traducción: {traduccion.get('terminos_mapeados', {})}")

            consulta_lower = consulta_usuario.lower()

            # === 3. CONSULTAS GENERALES: PASAR DIRECTO AL LLM ===
            consultas_generales = [
                'total personas', 'cuántas personas', 'número de personas', 'cuántas hay',
                'personas por edad', 'distribución por edad', 'por edad',
                'personas por sexo', 'distribución por sexo', 'por sexo',
                'por edad y sexo', 'edad y sexo', 'distribución por edad y sexo'
            ]
            if any(phrase in consulta_lower for phrase in consultas_generales):
                print("CONSULTA GENERAL DETECTADA → pasando al LLM sin validación estricta")
                return self.procesar_consulta(consulta_usuario)

            # === 4. VALIDACIÓN (solo si no es general) ===
            validacion = self.analizador.validar_variables_mejorado(traduccion)
            
            if not validacion['ejecutable']:
                if validacion['variables_invalidas']:
                    return self._generar_respuesta_variables_invalidas(validacion, consulta_usuario)
                else:
                    return self._generar_sugerencias_contextuales_mejoradas(consulta_usuario)

            # === 5. SI PASA: PROCESAR CON LLM ===
            print("Consulta válida → procesando con LLM...")
            return self.procesar_consulta(consulta_usuario)
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return self._generar_respuesta_error_amigable(e, consulta_usuario)

# ============================================================================
# 5. GENERADOR DE TABLAS (MÓDULO COMPLEMENTARIO)
# ============================================================================
class GeneradorTablas:
    """Genera tablas formateadas a partir de resultados de análisis"""
    
    def __init__(self):
        self.estilos_disponibles = ['markdown', 'html', 'pretty', 'grid']
        # Inyectar esta dependencia en el analizador
        # Nota: Esto se hace en main()
    
    def generar_tabla_desde_analisis(self, resultado_analisis: Dict, titulo: str = "", 
                                   formato: str = "markdown") -> str:
        """Convierte resultados de análisis en tablas formateadas"""
        
        try:
            if "error" in resultado_analisis:
                return f"❌ Error: {resultado_analisis['error']}"
            
            # TABLA PARA ANÁLISIS GEOGRÁFICO
            if "analisis_geografico" in resultado_analisis:
                geo_data = resultado_analisis["analisis_geografico"]
                
                if "top_ubicaciones" in geo_data:
                    datos_tabla = []
                    for i, (nombre, conteo, porcentaje) in enumerate(zip(
                        geo_data["top_ubicaciones"]["nombres"],
                        geo_data["top_ubicaciones"]["conteos"], 
                        geo_data["top_ubicaciones"]["porcentajes"]
                    )):
                        datos_tabla.append({
                            "Rank": i + 1,
                            "Ubicación": nombre,
                            "Personas": conteo,
                            "Porcentaje": f"{porcentaje}%"
                        })
                    
                    return self._formatear_tabla(datos_tabla, titulo or "Ranking Geográfico", formato)
            
            # TABLA PARA PERFIL DEMOGRÁFICO
            if "perfil_demografico" in resultado_analisis:
                demo_data = resultado_analisis["perfil_demografico"]
                datos_tabla = []
                
                if "distribucion_sexo" in demo_data:
                    total_personas = demo_data.get("total_personas", 0)
                    if total_personas == 0: # Fallback por si falta
                        total_personas = sum(demo_data["distribucion_sexo"].values())

                    for sexo, conteo in demo_data["distribucion_sexo"].items():
                        datos_tabla.append({
                            "Sexo": sexo,
                            "Cantidad": conteo,
                            "Porcentaje": f"{(conteo / total_personas * 100):.1f}%" if total_personas > 0 else "0.0%"
                        })
                
                return self._formatear_tabla(datos_tabla, titulo or "Distribución por Sexo", formato)
            
            # TABLA GENÉRICA PARA MÉTRICAS
            metricas_data = []
            if "metricas_principales" in resultado_analisis:
                for clave, valor in resultado_analisis["metricas_principales"].items():
                    metricas_data.append({"Métrica": clave.replace("_", " ").title(), "Valor": valor})
                
                return self._formatear_tabla(metricas_data, titulo or "Métricas Principales", formato)
            
            return "ℹ️ No hay datos tabulables en el resultado"
            
        except Exception as e:
            return f"❌ Error generando tabla: {str(e)}"
    
    def _formatear_tabla(self, datos: List[Dict], titulo: str, formato: str) -> str:
        """Formatea datos en tabla según formato especificado"""
        if not datos:
            return "No hay datos para mostrar"
        
        df = pd.DataFrame(datos)
        
        if formato == "markdown":
            tabla = df.to_markdown(index=False)
            return f"**{titulo}**\n\n{tabla}"
        
        elif formato == "html":
            tabla = df.to_html(index=False, classes='table table-striped')
            return f"<h3>{titulo}</h3>{tabla}"
        
        elif formato == "pretty":
            try:
                from tabulate import tabulate
                return tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
            except ImportError:
                print("⚠️  'tabulate' no instalado. Usando markdown.")
                return df.to_markdown(index=False)
        
        else:  # grid por defecto
            try:
                from tabulate import tabulate
                return tabulate(df, headers='keys', tablefmt='grid', showindex=False)
            except ImportError:
                print("⚠️  'tabulate' no instalado. Usando markdown.")
                return df.to_markdown(index=False)


# ============================================================================
# 6. FUNCIÓN PRINCIPAL Y EJECUCIÓN
# ============================================================================

def main():
    """Función principal para ejecutar el sistema completo con monitoreo de integridad"""
    print("🚀 INICIANDO SISTEMA DE ANÁLISIS MODULAR - CAPACIDADES DE BRECHAS")
    print("=" * 70)
    
    # ========================================================================
    # FASE 1: CARGA Y VALIDACIÓN DE DATOS
    # ========================================================================
    print("\n📋 FASE 1: CARGA Y VALIDACIÓN DE DATOS")
    print("-" * 70)
    
    try:
        integrator = DataIntegrator()
        # Intentar detectar ruta automáticamente primero
        df_completo = integrator.cargar_y_unir_datasets(ruta_base=None) 
    except FileNotFoundError as e_find:
        print(f"{e_find}")
        print("\n⚠️  No se pudo detectar la ruta automática. Reintentando con ruta '01_data/'...")
        try:
            # Fallback a la ruta original si la detección falla
            df_completo = integrator.cargar_y_unir_datasets("01_data/")
        except FileNotFoundError as e_final:
            print(f"❌ ERROR FINAL: {e_final}")
            print("Asegúrate de que la carpeta '01_data' exista en el directorio correcto.")
            return
    except Exception as e:
        print(f"❌ Error fatal durante la carga de datos: {e}")
        return

    
    # Auditoría de integridad
    metricas_auditoria = integrator.auditar_integridad_completa()
    
    # ========================================================================
    # FASE 2: MONITOREO Y COMPARACIÓN DE INTEGRIDAD
    # ========================================================================
    print("\n📊 FASE 2: MONITOREO DE INTEGRIDAD")
    print("-" * 70)
    
    # Comparar con reportes anteriores si existen
    resultado_comparacion = integrator.comparar_reportes_integridad()
    
    # Generar reporte temporal
    integrator.generar_reporte_temporal()
    
    # Limpiar reportes antiguos (opcional - mantiene últimos 30 días)
    integrator.limpiar_reportes_antiguos(dias_retencion=30)
    
    # ========================================================================
    # FASE 3: ANÁLISIS EXPLORATORIO E INYECCIÓN DE DEPENDENCIAS
    # ========================================================================
    print("\n🔍 FASE 3: ANÁLISIS EXPLORATORIO")
    print("-" * 70)
    
    analizador = AnalizadorUnidimensional(df_completo, API_KEY)
    
    # Inyectar dependencia del generador de tablas (que fue definido después)
    analizador.generador_tablas = GeneradorTablas()
    
    ubicaciones = analizador.explorar_ubicaciones_disponibles(top_n=5)
    
    print("\n📍 UBICACIONES DISPONIBLES:")
    print(f"  • Colonias: {ubicaciones['colonias_mas_pobladas']['total_colonias']}")
    print(f"  • AGEBs: {ubicaciones['agebs_mas_poblados']['total_agebs']}")
    
    print("\n🏙️ TOP 3 COLONIAS MÁS POBLADAS:")
    for i, (colonia, count) in enumerate(list(ubicaciones['colonias_mas_pobladas']['top_colonias'].items())[:3], 1):
        print(f"  {i}. {colonia}: {count:,} personas")
    
    # ========================================================================
    # FASE 4: INICIALIZACIÓN DEL AGENTE
    # ========================================================================
    print("\n🤖 FASE 4: INICIALIZACIÓN DEL AGENTE")
    print("-" * 70)
    
    try:
        API_KEY = get_api_key()
        agente = AgenteAnaliticoLLM(df_completo, API_KEY)
        
        # Inyectar el analizador ya creado (con el generador de tablas)
        # en el agente para que ambos usen la MISMA instancia.
        agente.analizador = analizador
        
        print("✅ Agente inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando agente: {str(e)}")
        return
    
    # ========================================================================
    # FASE 5: INTERFAZ INTERACTIVA
    # ========================================================================
    print("\n" + "="*70)
    print("💬 AGENTE CON CAPACIDADES DE BRECHAS - Pregunta en lenguaje natural")
    print("="*70)
    
    print("\n🎯 **EJEMPLOS DE CONSULTAS DISPONIBLES:**")
    print("  📊 Análisis de Brechas:")
    print("     - 'Niños con múltiples carencias'")
    print("     - 'Adultos mayores sin pensión'")
    print("     - 'Personas con carencia de salud sin programas'")
    print("  ")
    print("  📈 Análisis Comparativos:")
    print("     - 'Comparar brechas entre pensiones adultos mayores y mujeres'")
    print("     - 'Mujeres elegibles para pensión que no la reciben'")
    print("  ")
    print("  🗺️ Análisis Geográfico:")
    print("     - 'Distribuir elegibles por AGEB'")
    print("     - 'Cobertura de programas por colonia'")
    print("  ")
    print("  📋 Comandos Especiales:")
    print("     - '/ayuda' → Sugerencias de consultas")
    print("     - '/integridad' → Mostrar estado de integridad")
    print("     - '/timeline' → Mostrar evolución temporal")
    print("     - '/salir' → Terminar sesión")
    print("-" * 70)
    
    # Variables para seguimiento de sesión
    consultas_procesadas = 0
    
    while True:
        try:
            consulta = input("\n🗣️ Tú: ").strip()
            
            # Comandos especiales
            if consulta.lower() in ['/salir', '/exit', 'salir', 'exit']:
                print("👋 ¡Hasta luego!")
                print(f"\n📊 Resumen de sesión:")
                print(f"  • Consultas procesadas: {consultas_procesadas}")
                print(f"  • Hogares en dataset: {metricas_auditoria['total_hogares']:,}")
                print(f"  • Personas analizadas: {metricas_auditoria['total_personas']:,}")
                break
            
            elif consulta.lower() in ['/ayuda', 'ayuda', '/sugerencias']:
                print("\n🎯 **CONSULTAS DE BRECHAS SUGERIDAS:**")
                print("• 'Niños con múltiples carencias'")
                print("• 'Adultos mayores de 65 años sin pensión'")
                print("• 'Personas con carencia de salud sin programas'")
                print("• 'Mujeres elegibles para pensión que no la reciben'")
                print("• 'Comparar brechas entre becas Benito Juárez y Rita Cetina'")
                print("• 'Niños 0-12 años con rezago educativo sin becas'")
                print("• 'Distribución de hogares por AGEB'")
                print("• 'Top colonias con mayor carencia de seguridad social'")
                continue
            
            elif consulta.lower() == '/integridad':
                print("\n📊 ESTADO ACTUAL DE INTEGRIDAD:")
                print(f"  ✓ Total personas: {metricas_auditoria['total_personas']:,}")
                print(f"  ✓ Total hogares: {metricas_auditoria['total_hogares']:,}")
                print(f"  ✓ Personas/hogar promedio: {metricas_auditoria['personas_por_hogar_promedio']:.2f}")
                print(f"  ✓ Integridad hogares: {'✅ PERFECTA' if metricas_auditoria['integridad_hogares'] else '⚠️ CON DISCREPANCIAS'}")
                
                if resultado_comparacion:
                    print(f"\n📈 CAMBIOS DESDE ÚLTIMO REPORTE:")
                    print(f"  • Hogares nuevos: {len(resultado_comparacion.get('nuevos', [])):,}")
                    print(f"  • Hogares recuperados: {len(resultado_comparacion.get('recuperados', [])):,}")
                    print(f"  • Cambio neto: {resultado_comparacion.get('cambio_neto', 0):+d}")
                continue
            
            elif consulta.lower() == '/timeline':
                print("\n📊 TIMELINE DE INTEGRIDAD:")
                integrator.generar_reporte_temporal()
                continue
            
            elif not consulta:
                continue
            
            # Procesar consulta normal
            print("\n⏳ Procesando...")
            respuesta = agente.procesar_consulta_mejorado(consulta)
            print(f"\n🤖 Agente: {respuesta}")
            consultas_procesadas += 1
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Sesión interrumpida por usuario")
            break
        
        except Exception as e:
            print(f"\n❌ Error procesando consulta: {str(e)}")
            print("💡 Intenta reformular tu pregunta o usa /ayuda para sugerencias")

if __name__ == "__main__":
    main()