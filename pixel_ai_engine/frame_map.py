"""
Pixel AI Engine - Unified Frame & Semantic Grid Map (frame_map.py)
Define el mapeo unico compartido para formato 8x12 (96 frames) y 16x4 (64 frames):
coordenadas, indices, filas, columnas, orientacion cardinal, acciones y ciclos de animacion.
"""

from typing import Dict, List, Any, Optional

# ============================================================================
# FORMATO PRINCIPAL CANONICO: 8x12 (96 FRAMES - 1024x1536 px, celdas 128x128)
# ============================================================================

FORMAT_8X12 = {
    "format_id": "8x12",
    "name": "Canónico 8x12 (96 Poses)",
    "cols": 8,
    "rows": 12,
    "total_frames": 96,
    "cell_width": 128,
    "cell_height": 128,
    "canvas_width": 1024,
    "canvas_height": 1536,
    "test_pose_indices": [0, 16, 32, 92],  # Frente, Perfil Este, Espalda Norte, Celebración
    "row_definitions": [
        {"row": 0, "id": "walk_south", "name": "Caminata Sur (Frente)", "direction": "sur", "action": "caminar"},
        {"row": 1, "id": "walk_southeast", "name": "Caminata Sureste (Diag. Frontal Der.)", "direction": "sureste", "action": "caminar"},
        {"row": 2, "id": "walk_east", "name": "Caminata Este (Lateral Derecho)", "direction": "este", "action": "caminar"},
        {"row": 3, "id": "walk_northeast", "name": "Caminata Noreste (Diag. Trasera Der.)", "direction": "noreste", "action": "caminar"},
        {"row": 4, "id": "walk_north", "name": "Caminata Norte (Espalda)", "direction": "norte", "action": "caminar"},
        {"row": 5, "id": "walk_northwest", "name": "Caminata Noroeste (Diag. Trasera Izq.)", "direction": "noroeste", "action": "caminar"},
        {"row": 6, "id": "walk_west", "name": "Caminata Oeste (Lateral Izquierdo)", "direction": "oeste", "action": "caminar"},
        {"row": 7, "id": "walk_southwest", "name": "Caminata Suroeste (Diag. Frontal Izq.)", "direction": "suroeste", "action": "caminar"},
        {"row": 8, "id": "cook_bowl", "name": "Cocina - Preparación con Bowl", "direction": "sur", "action": "cocinar"},
        {"row": 9, "id": "cook_station", "name": "Cocina - Estación de Trabajo", "direction": "este", "action": "cocinar"},
        {"row": 10, "id": "think_carry", "name": "Pensar (cols 0-3) y Cargar Caja (cols 4-7)", "direction": "sur", "action": "pensar_cargar"},
        {"row": 11, "id": "serve_celebrate", "name": "Servir Plato (cols 0-3) y Celebración (cols 4-7)", "direction": "sur", "action": "servir_celebrar"},
    ],
    "animation_clips": {
        "caminar_sur": list(range(0, 8)),
        "caminar_sureste": list(range(8, 16)),
        "caminar_este": list(range(16, 24)),
        "caminar_noreste": list(range(24, 32)),
        "caminar_norte": list(range(32, 40)),
        "caminar_noroeste": list(range(40, 48)),
        "caminar_oeste": list(range(48, 56)),
        "caminar_suroeste": list(range(56, 64)),
        "cocinar_bowl": list(range(64, 72)),
        "cocinar_estacion": list(range(72, 80)),
        "pensar": list(range(80, 84)),
        "cargar_caja": list(range(84, 88)),
        "servir_plato": list(range(88, 92)),
        "celebrar": list(range(92, 96)),
        "idle_reposo": [0, 80],
    }
}

# ============================================================================
# FORMATO LEGACY COMPATIBLE: 16x4 (64 FRAMES - 682x2048 px, celdas 170x128)
# ============================================================================

FORMAT_16X4 = {
    "format_id": "16x4",
    "name": "Base Legacy 16x4 (64 Poses)",
    "cols": 4,
    "rows": 16,
    "total_frames": 64,
    "cell_width": 170,
    "cell_height": 128,
    "canvas_width": 682,
    "canvas_height": 2048,
    "test_pose_indices": [0, 16, 8, 48],
    "row_definitions": [
        {"row": 0, "id": "walk_south_idle", "name": "Frente - Reposo/Idle", "direction": "sur", "action": "idle"},
        {"row": 1, "id": "walk_south_steps", "name": "Frente - Pasos", "direction": "sur", "action": "caminar"},
        {"row": 2, "id": "walk_north_idle", "name": "Espalda - Reposo/Idle", "direction": "norte", "action": "idle"},
        {"row": 3, "id": "walk_north_steps", "name": "Espalda - Pasos", "direction": "norte", "action": "caminar"},
        {"row": 4, "id": "walk_east_idle", "name": "Lateral Der. - Reposo/Idle", "direction": "este", "action": "idle"},
        {"row": 5, "id": "walk_east_steps", "name": "Lateral Der. - Pasos", "direction": "este", "action": "caminar"},
        {"row": 6, "id": "walk_west_idle", "name": "Lateral Izq. - Reposo/Idle", "direction": "oeste", "action": "idle"},
        {"row": 7, "id": "walk_west_steps", "name": "Lateral Izq. - Pasos", "direction": "oeste", "action": "caminar"},
        {"row": 8, "id": "diag_front_right", "name": "Diagonal Frontal Derecha", "direction": "sureste", "action": "caminar"},
        {"row": 9, "id": "diag_front_left", "name": "Diagonal Frontal Izquierda", "direction": "suroeste", "action": "caminar"},
        {"row": 10, "id": "diag_back_right", "name": "Diagonal Trasera Derecha", "direction": "noreste", "action": "caminar"},
        {"row": 11, "id": "diag_back_left", "name": "Diagonal Trasera Izquierda", "direction": "noroeste", "action": "caminar"},
        {"row": 12, "id": "idle_breath", "name": "Reposo y Respiración", "direction": "sur", "action": "idle"},
        {"row": 13, "id": "static_actions", "name": "Acciones Estáticas", "direction": "sur", "action": "accion"},
        {"row": 14, "id": "hand_variations", "name": "Variaciones de Manos", "direction": "sur", "action": "manos"},
        {"row": 15, "id": "close_expressions", "name": "Expresiones de Cierre", "direction": "sur", "action": "expresion"},
    ],
    "animation_clips": {
        "caminar_frente": list(range(0, 8)),
        "caminar_espalda": list(range(8, 16)),
        "caminar_lateral_der": list(range(16, 24)),
        "caminar_lateral_izq": list(range(24, 32)),
        "diagonal_frontal_der": list(range(32, 36)),
        "diagonal_frontal_izq": list(range(36, 40)),
        "diagonal_trasera_der": list(range(40, 44)),
        "diagonal_trasera_izq": list(range(44, 48)),
        "reposo": list(range(48, 52)),
    }
}


def get_frame_semantic_info(frame_idx: int, format_id: str = "8x12") -> Dict[str, Any]:
    """
    Devuelve los metadatos semanticos exactos de cualquier frame segun su celda.
    Resuelve con precision direccion cardinal, accion, fase y descripcion de prompt.
    """
    cfg = FORMAT_8X12 if format_id == "8x12" else FORMAT_16X4
    cols = cfg["cols"]
    rows = cfg["rows"]
    
    if frame_idx < 0 or frame_idx >= cfg["total_frames"]:
        raise ValueError(f"frame_idx {frame_idx} fuera de rango para formato {format_id} (0..{cfg['total_frames']-1})")
        
    r = frame_idx // cols
    c = frame_idx % cols
    
    if format_id == "8x12":
        row_def = cfg["row_definitions"][r]
        direction = row_def["direction"]
        action = row_def["action"]
        
        # Sub-division de filas hibridas 10 y 11
        if r == 10:
            if c < 4:
                action = "pensar"
                action_desc = "pensando con la mano en la barbilla"
                sub_phase = f"pensar_fase_{c+1}"
            else:
                action = "cargar_caja"
                action_desc = "cargando una caja con ambas manos al frente"
                sub_phase = f"cargar_fase_{c-3}"
        elif r == 11:
            if c < 4:
                action = "servir_plato"
                action_desc = "sosteniendo y sirviendo un plato de comida"
                sub_phase = f"servir_fase_{c+1}"
            else:
                action = "celebrar"
                action_desc = "celebrando con las manos arriba con entusiasmo"
                sub_phase = f"celebrar_fase_{c-3}"
        elif r in (8, 9):
            action_desc = "cocinando con utensilios de cocina"
            sub_phase = f"cocina_paso_{c+1}"
        else:
            action_desc = f"caminata hacia el {direction}"
            sub_phase = "idle_pie_apoyado" if c == 0 else f"paso_{c}"
            
        is_test = frame_idx in cfg["test_pose_indices"]
        
        return {
            "index": frame_idx,
            "row": r,
            "col": c,
            "format": format_id,
            "direction": direction,
            "action": action,
            "action_desc": action_desc,
            "sub_phase": sub_phase,
            "is_test_pose": is_test,
            "x0": int(round(c * cfg["canvas_width"] / cols)),
            "y0": int(round(r * cfg["canvas_height"] / rows)),
            "x1": int(round((c + 1) * cfg["canvas_width"] / cols)),
            "y1": int(round((r + 1) * cfg["canvas_height"] / rows)),
        }
    else:
        row_def = cfg["row_definitions"][r]
        return {
            "index": frame_idx,
            "row": r,
            "col": c,
            "format": format_id,
            "direction": row_def["direction"],
            "action": row_def["action"],
            "action_desc": row_def["name"],
            "sub_phase": f"fase_{c+1}",
            "is_test_pose": frame_idx in cfg["test_pose_indices"],
            "x0": int(round(c * cfg["canvas_width"] / cols)),
            "y0": int(round(r * cfg["canvas_height"] / rows)),
            "x1": int(round((c + 1) * cfg["canvas_width"] / cols)),
            "y1": int(round((r + 1) * cfg["canvas_height"] / rows)),
        }


def get_all_frame_mappings(format_id: str = "8x12") -> List[Dict[str, Any]]:
    """Devuelve el mapa semantico completo de todos los frames del formato."""
    cfg = FORMAT_8X12 if format_id == "8x12" else FORMAT_16X4
    return [get_frame_semantic_info(i, format_id) for i in range(cfg["total_frames"])]
