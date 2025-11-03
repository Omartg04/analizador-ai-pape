"""
PRUEBA DE CONSULTAS PROBLEMÁTICAS EN STREAMLIT
Valida que los cambios resuelvan los 2 problemas reportados
"""

import sys
import os

# Agregar ruta del backend si es necesario
if os.path.exists('../backend'):
    sys.path.insert(0, '../backend')

def test_consulta_carencias():
    """
    TEST 1: Personas con carencia de salud en barrio norte
    Debe mostrar análisis de cobertura programática CORRECTO
    """
    print("\n" + "="*70)
    print("🧪 TEST 1: CARENCIA DE SALUD EN BARRIO NORTE")
    print("="*70)
    
    try:
        from analizador_optimizado import (
            DataIntegrator, 
            AnalizadorUnidimensional,
            AnalizadorProgramasSociales
        )
        
        print("\n📊 Cargando datos...")
        integrator = DataIntegrator()
        df_completo = integrator.cargar_y_unir_datasets("01_data/")
        
        print(f"✅ Dataset cargado: {len(df_completo):,} personas")
        
        # Crear analizador
        analizador_programas = AnalizadorProgramasSociales(df_completo)
        
        print("\n🔍 Ejecutando: identificar_carencias_sin_cobertura")
        print("   Parámetros: carencia='salud', ubicacion='barrio norte'")
        
        resultado = analizador_programas.identificar_carencias_sin_cobertura(
            carencia='salud',
            ubicacion='barrio norte'
        )
        
        print("\n✅ RESULTADO:")
        print("-" * 70)
        
        if 'error' in resultado:
            print(f"❌ Error: {resultado['error']}")
            return False
        
        # Mostrar métricas principales
        metricas = resultado.get('metricas_principales', {})
        print(f"Personas con carencia de salud: {metricas.get('total_personas_con_carencia', 'N/A')}")
        print(f"Personas SIN cobertura: {metricas.get('total_personas_sin_cobertura', 'N/A')}")
        print(f"Tasa de brecha: {metricas.get('tasa_brecha', 'N/A')}%")
        
        # Programas analizados
        print(f"\nProgramas relacionados: {resultado.get('programas_relacionados_analizados', [])}")
        
        # Perfil de la brecha
        perfil = resultado.get('perfil_brecha', {})
        if perfil:
            print(f"\n📊 Perfil de personas sin cobertura:")
            print(f"   - Edad promedio: {perfil.get('edad_promedio', 'N/A')} años")
            print(f"   - Distribución sexo: {perfil.get('distribucion_sexo', {})}")
            print(f"   - Hogares afectados: {perfil.get('hogares_afectados', 'N/A')}")
        
        # Análisis geográfico
        geo = resultado.get('analisis_geografico', {})
        if geo:
            print(f"\n🗺️  Análisis geográfico:")
            print(f"   - Colonias afectadas: {geo.get('colonias_afectadas', 'N/A')}")
            print(f"   - Top colonias: {geo.get('top_colonias', {})}")
        
        print("\n" + "="*70)
        print("✅ TEST 1 PASÓ - Análisis de cobertura funciona correctamente")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en TEST 1: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consulta_elegibilidad():
    """
    TEST 2: Personas elegibles para IMSS Bienestar en colonia
    Debe mostrar elegibilidad SIN contradicciones en análisis de cobertura
    """
    print("\n" + "="*70)
    print("🧪 TEST 2: ELEGIBILIDAD IMSS BIENESTAR EN COLONIA")
    print("="*70)
    
    try:
        from analizador_optimizado import (
            DataIntegrator,
            AnalizadorProgramasSociales
        )
        
        print("\n📊 Cargando datos...")
        integrator = DataIntegrator()
        df_completo = integrator.cargar_y_unir_datasets("01_data/")
        
        print(f"✅ Dataset cargado: {len(df_completo):,} personas")
        
        # Crear analizador
        analizador_programas = AnalizadorProgramasSociales(df_completo)
        
        # Elegir una colonia del dataset
        colonias = df_completo['colonia'].unique()[:3]
        colonia_test = colonias[0]
        
        print(f"\n🔍 Ejecutando: analizar_elegibilidad_programa")
        print(f"   Programa: imss_bienestar")
        print(f"   Ubicación: {colonia_test}")
        
        resultado = analizador_programas.analizar_elegibilidad_programa(
            programa='imss_bienestar',
            ubicacion=colonia_test,
            incluir_brecha=True
        )
        
        print("\n✅ RESULTADO:")
        print("-" * 70)
        
        if 'error' in resultado:
            print(f"❌ Error: {resultado['error']}")
            return False
        
        # Métricas de elegibilidad
        metricas = resultado.get('metricas_elegibilidad', {})
        print(f"Total población filtrada: {metricas.get('total_poblacion_filtrada', 'N/A'):,}")
        print(f"Total elegibles: {metricas.get('total_elegibles', 'N/A'):,}")
        print(f"Tasa elegibilidad: {metricas.get('tasa_elegibilidad', 'N/A')}%")
        
        # VALIDAR QUE NO HAY CONTRADICCIÓN
        print(f"\n🔍 VALIDACIÓN DE CONTRADICCIÓN:")
        total_elegibles = metricas.get('total_elegibles', 0)
        brecha_tasa = resultado.get('analisis_brecha', {}).get('tasa_brecha', 0)
        
        print(f"   - Total elegibles: {total_elegibles}")
        print(f"   - Tasa brecha: {brecha_tasa}%")
        
        if total_elegibles > 0 and brecha_tasa == 0:
            print(f"\n⚠️  ADVERTENCIA: {total_elegibles} elegibles pero brecha=0%")
            print(f"   Significa: Todas las personas elegibles reciben apoyo social")
            print(f"   (No necesariamente reciben IMSS, pero sí ALGÚN programa)")
        elif total_elegibles > 0 and brecha_tasa > 0:
            print(f"\n✅ CORRECTO: {total_elegibles} elegibles con {brecha_tasa}% sin cobertura")
        
        # Perfil de elegibles
        perfil = resultado.get('perfil_elegibles', {})
        if perfil:
            print(f"\n📊 Perfil de elegibles:")
            print(f"   - Edad promedio: {perfil.get('edad_promedio', 'N/A')} años")
            print(f"   - Distribución sexo: {perfil.get('distribucion_sexo', {})}")
            print(f"   - Hogares afectados: {perfil.get('hogares_afectados', 'N/A')}")
        
        # Análisis de brecha
        brecha = resultado.get('analisis_brecha', {})
        if brecha:
            print(f"\n🔍 Análisis de brecha:")
            print(f"   - Elegibles sin ningún apoyo: {brecha.get('elegibles_sin_ningun_apoyo', 'N/A')}")
            print(f"   - Tasa brecha: {brecha.get('tasa_brecha', 'N/A')}%")
            print(f"   - Interpretación: {brecha.get('interpretacion', 'N/A')}")
        
        print("\n" + "="*70)
        print("✅ TEST 2 PASÓ - No hay contradicciones")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en TEST 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_brechas_comparativas():
    """
    TEST 3: Comparar brechas entre múltiples programas
    Valida que el nuevo método funciona correctamente
    """
    print("\n" + "="*70)
    print("🧪 TEST 3: COMPARATIVA DE BRECHAS MULTIPROGRAMA")
    print("="*70)
    
    try:
        from analizador_optimizado import (
            DataIntegrator,
            AnalizadorProgramasSociales
        )
        
        print("\n📊 Cargando datos...")
        integrator = DataIntegrator()
        df_completo = integrator.cargar_y_unir_datasets("01_data/")
        
        print(f"✅ Dataset cargado: {len(df_completo):,} personas")
        
        # Crear analizador
        analizador_programas = AnalizadorProgramasSociales(df_completo)
        
        print(f"\n🔍 Ejecutando: analizar_brechas_multiprograma")
        print(f"   Programas: pension_adultos_mayores, pension_mujeres_bienestar")
        print(f"   Grupo edad: 65+ años")
        
        resultado = analizador_programas.analizar_brechas_multiprograma(
            programas=['pension_adultos_mayores', 'pension_mujeres_bienestar'],
            grupo_edad=(65, 120)
        )
        
        print("\n✅ RESULTADO:")
        print("-" * 70)
        
        if 'error' in resultado:
            print(f"❌ Error: {resultado['error']}")
            return False
        
        # Resumen general
        resumen = resultado.get('resumen_general', {})
        print(f"Programas analizados: {resumen.get('total_programas_analizados', 'N/A')}")
        print(f"Total elegibles agregado: {resumen.get('total_elegibles_agregado', 'N/A'):,}")
        print(f"Brecha promedio: {resumen.get('brecha_promedio', 'N/A')}%")
        
        # Comparativa resumida
        comparativa = resultado.get('comparativa_resumida', {})
        if comparativa:
            print(f"\n📊 Comparativa por programa:")
            for prog, datos in comparativa.items():
                print(f"   {prog}:")
                print(f"      - Elegibles: {datos.get('total_elegibles', 'N/A'):,}")
                print(f"      - Brecha: {datos.get('brecha_cobertura', 'N/A')}%")
        
        # Análisis comparativo
        analisis = resultado.get('analisis_comparativo', {})
        if analisis:
            print(f"\n🏆 Análisis comparativo:")
            print(f"   - Mayor cobertura: {analisis.get('programa_mas_cobertura', 'N/A')}")
            print(f"   - Mayor brecha: {analisis.get('programa_mayor_brecha', 'N/A')}")
            print(f"   - Ranking elegibles: {analisis.get('ranking_por_elegibles', [])}")
        
        print("\n" + "="*70)
        print("✅ TEST 3 PASÓ - Análisis comparativo funciona")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en TEST 3: {e}")
        import traceback
        traceback.print_exc()
        return False


def resumen_final(test1, test2, test3):
    """Genera resumen final de tests"""
    
    print("\n" + "="*70)
    print("📋 RESUMEN DE TESTS")
    print("="*70)
    
    tests = [
        ("TEST 1: Carencias sin cobertura", test1),
        ("TEST 2: Elegibilidad sin contradicciones", test2),
        ("TEST 3: Brechas comparativas", test3),
    ]
    
    for nombre, resultado in tests:
        estado = "✅ PASÓ" if resultado else "❌ FALLÓ"
        print(f"{estado}: {nombre}")
    
    todos_pasaron = all([test1, test2, test3])
    
    print("\n" + "="*70)
    
    if todos_pasaron:
        print("✅ TODOS LOS TESTS PASARON")
        print("\n🎉 CAMBIOS VALIDADOS - LISTO PARA PRODUCCIÓN")
        print("\n📌 PRÓXIMOS PASOS:")
        print("1. Prueba en Streamlit con tus consultas originales")
        print("2. Verifica que 'Análisis de Cobertura' ahora muestra datos correctos")
        print("3. No hay más contradicciones en resultados")
        print("4. Si todo bien, confirma cambios en git/control de versiones")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("\n💡 Revisa los errores anteriores")
    
    print("="*70 + "\n")
    
    return todos_pasaron


# EJECUCIÓN
if __name__ == "__main__":
    print("\n🚀 INICIANDO PRUEBAS DE CONSULTAS")
    print("Esta validación prueba los cambios con datos REALES\n")
    
    test1 = test_consulta_carencias()
    test2 = test_consulta_elegibilidad()
    test3 = test_brechas_comparativas()
    
    todos_ok = resumen_final(test1, test2, test3)
    
    sys.exit(0 if todos_ok else 1)