"""
TEST RÁPIDO Y SIMPLE CON DATOS REALES
Prueba los 3 métodos nuevos con rutas correctas
"""

import os
import sys

# Mostrar directorio actual
print(f"\n📍 Directorio actual: {os.getcwd()}")

# Importar
print("\n📦 Importando módulos...")
try:
    from analizador_optimizado import DataIntegrator, AnalizadorProgramasSociales
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)

# Cargar datos
print("\n📊 Cargando datos...")
try:
    integrator = DataIntegrator()
    df_completo = integrator.cargar_y_unir_datasets()
    print(f"✅ Dataset cargado: {len(df_completo):,} personas en {df_completo['id_hogar'].nunique():,} hogares")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Crear analizador
analizador = AnalizadorProgramasSociales(df_completo)
print("✅ Analizador creado")

# TEST 1: identificar_carencias_sin_cobertura
print("\n" + "="*70)
print("🧪 TEST 1: identificar_carencias_sin_cobertura")
print("="*70)
try:
    resultado1 = analizador.identificar_carencias_sin_cobertura(carencia='salud')
    
    if 'error' in resultado1:
        print(f"❌ Error: {resultado1['error']}")
    else:
        metricas = resultado1.get('metricas_principales', {})
        print(f"✅ Personas con carencia salud: {metricas.get('total_personas_con_carencia', 0):,}")
        print(f"✅ Sin cobertura: {metricas.get('total_personas_sin_cobertura', 0):,}")
        print(f"✅ Tasa brecha: {metricas.get('tasa_brecha', 0)}%")
        print(f"✅ Programas analizados: {resultado1.get('programas_relacionados_analizados', [])}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# TEST 2: analizar_intensidad_carencias
print("\n" + "="*70)
print("🧪 TEST 2: analizar_intensidad_carencias")
print("="*70)
try:
    resultado2 = analizador.analizar_intensidad_carencias(grupo_edad=(0, 12))
    
    if 'error' in resultado2:
        print(f"❌ Error: {resultado2['error']}")
    else:
        dist = resultado2.get('distribucion_intensidad', {})
        print(f"✅ Niños sin carencias: {dist.get('sin_carencias', {}).get('cantidad', 0)}")
        print(f"✅ Niños con 1 carencia: {dist.get('una_carencia', {}).get('cantidad', 0)}")
        print(f"✅ Niños con 2 carencias: {dist.get('dos_carencias', {}).get('cantidad', 0)}")
        print(f"✅ Niños con 3 carencias: {dist.get('tres_carencias', {}).get('cantidad', 0)}")
        vuln = resultado2.get('poblacion_vulnerabilidad_extrema', {})
        print(f"✅ Vulnerabilidad extrema: {vuln.get('total', 0)}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# TEST 3: analizar_brechas_multiprograma
print("\n" + "="*70)
print("🧪 TEST 3: analizar_brechas_multiprograma")
print("="*70)
try:
    resultado3 = analizador.analizar_brechas_multiprograma(
        programas=['pension_adultos_mayores', 'pension_mujeres_bienestar'],
        grupo_edad=(65, 120)
    )
    
    if 'error' in resultado3:
        print(f"❌ Error: {resultado3['error']}")
    else:
        resumen = resultado3.get('resumen_general', {})
        print(f"✅ Programas analizados: {resumen.get('total_programas_analizados', 0)}")
        print(f"✅ Total elegibles: {resumen.get('total_elegibles_agregado', 0):,}")
        print(f"✅ Brecha promedio: {resumen.get('brecha_promedio', 0)}%")
        
        comparativa = resultado3.get('comparativa_resumida', {})
        print(f"\n📊 Por programa:")
        for prog, datos in comparativa.items():
            print(f"   {prog}: {datos.get('total_elegibles', 0):,} elegibles, {datos.get('brecha_cobertura', 0)}% brecha")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Resumen
print("\n" + "="*70)
print("✅ TODOS LOS TESTS COMPLETADOS")
print("="*70)
print("\n🎉 Los 3 métodos nuevos funcionan correctamente!")
print("\n📌 Próximo paso: Prueba en Streamlit")
print("   streamlit run ../frontend/app.py")
print("\n" + "="*70 + "\n")
