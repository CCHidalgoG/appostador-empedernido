# Jev (TypeSafe AI): guía de uso e integración

Esta guía explica cómo probar Jev, conectarlo desde una aplicación y usarlo junto a un agente de programación. Se basa en la [documentación oficial de TypeSafe](https://docs.typesafe.ai/introduction). Los nombres de paquetes y requisitos pueden cambiar; consulta los enlaces oficiales antes de actualizar una integración.

## 1. Qué es Jev

Jev es el modelo de decisiones estructuradas de TypeSafe (System One). Recibe un **estado** —texto, objeto o arreglo con el contexto que tu aplicación quiere evaluar— y una o varias **preguntas tipadas**. Devuelve valores estructurados que el código puede usar sin pedirle al modelo que genere JSON ni analizar texto libre.

No es un chatbot ni un modelo de autocompletado. No escribe código ni reemplaza al LLM que impulsa Claude Code, Codex, Cursor u otro coding agent. En cambio, una aplicación o agente que construyas puede llamar a Jev para tomar decisiones acotadas.

### Tipos de pregunta

| Tipo | Cuándo usarlo | Resultado |
| --- | --- | --- |
| `choice` | Elegir una opción de un conjunto cerrado (categoría, ruta, equipo) | Opción elegida, probabilidades por opción y confianza |
| `score` | Ubicar el estado en una escala ordenada definida por ti | Puntaje, distribución de probabilidades y confianza |
| `noul` | Evaluar una afirmación de sí/no | Valor entre 0 y 1 que representa la probabilidad de que sea verdadera |

Las preguntas de una llamada se evalúan en paralelo e independientemente contra el mismo estado. Formula preguntas atómicas y específicas; si una decisión combina varios factores, pregunta por cada factor y combínalos en tu código.

## 2. Probarlo sin escribir código

1. Abre el [Playground de TypeSafe](https://console.typesafe.ai/) e inicia sesión.
2. Pega un texto o estructura como estado.
3. Añade una pregunta `noul`, `choice` o `score`; prueba combinar varias.
4. Para usarlo desde tu aplicación, crea una API key en el dashboard.

No incluyas credenciales reales en documentación, código versionado ni frontend público.

## 3. Conexión mediante HTTP

La API de evaluación usa `POST https://api.typesafe.ai/v1/systemone`. Envía la clave como `Authorization: Bearer ...` y el cuerpo JSON con `state`, `model` y `questions`.

Guarda la clave como variable de entorno `TYPESAFE_API_KEY` en el entorno donde corre el backend. En desarrollo puede cargarse desde un `.env` local no versionado; en despliegues, usa el gestor de secretos del proveedor. No la uses en código que se entrega al navegador.

Ejemplo con cURL (requiere que `TYPESAFE_API_KEY` ya esté exportada):

```bash
curl -X POST https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"mensaje": "Me han cobrado dos veces. Necesito ayuda hoy."},
    "model": "jev-latest",
    "questions": {
      "area": {
        "type": "choice",
        "instructions": "¿Qué equipo debería atender este caso?",
        "criteria": {
          "facturacion": "Cobros, pagos y reembolsos",
          "tecnico": "Errores e integraciones",
          "otro": "Consultas que no encajan en las anteriores"
        }
      },
      "urgente": {
        "type": "noul",
        "instructions": "¿El mensaje indica que el problema requiere atención hoy?"
      },
      "prioridad": {
        "type": "score",
        "instructions": "¿Qué prioridad comunica el mensaje?",
        "criteria": ["Puede esperar", "Atender pronto", "Atender hoy"]
      }
    }
  }'
```

El objeto `answers` devuelve cada respuesta bajo la misma clave usada en `questions`. Choice y Score también incluyen probabilidades y confianza; la respuesta trae información de uso de tokens. Consulta el [contrato HTTP completo](https://docs.typesafe.ai/api) para campos, errores y límites.

## 4. Conectar desde una aplicación

Usa el SDK del lenguaje para conservar tipos y simplificar las llamadas. La clave se lee desde `TYPESAFE_API_KEY`.

### JavaScript / TypeScript

Requisito documentado: Node.js 20 o superior.

```bash
npm install @typesafe-ai/sdk
```

```ts
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient();
const response = await client.systemOne({
  state: { mensaje: "Me han cobrado dos veces. Necesito ayuda hoy." },
  questions: {
    categoria: choice("¿De qué trata el ticket?", {
      facturacion: "Cobros, pagos y reembolsos",
      tecnico: "Errores e integraciones",
      otro: "Otro tema",
    }),
  },
});

console.log(response.answers.categoria.choice);
```

La respuesta es inferida a partir de las preguntas y el SDK incluye tipos TypeScript. Consulta el [inicio rápido de JavaScript](https://docs.typesafe.ai/sdk/javascript).

### Python

Requisito documentado: Python 3.10 o superior.

```bash
pip install typesafe-sdk
# o: uv add typesafe-sdk
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={"mensaje": "Me han cobrado dos veces. Necesito ayuda hoy."},
        questions={
            "categoria": Choice(
                instructions="¿De qué trata el ticket?",
                criteria={
                    "facturacion": "Cobros, pagos y reembolsos",
                    "tecnico": "Errores e integraciones",
                    "otro": "Otro tema",
                },
            ),
            "urgente": Noul(
                instructions="¿El mensaje requiere atención hoy?",
            ),
        },
    )

print(response.answers["categoria"].choice)
print(response.answers["urgente"].noul)
```

El SDK también ofrece `AsyncTypeSafeClient`; mira el [inicio rápido de Python](https://docs.typesafe.ai/sdk/python) para la variante asíncrona y los tipos de respuesta.

## 5. Usar Jev con coding agents

Hay dos usos distintos:

1. **Pedirle al coding agent que integre Jev.** Instala la [skill oficial de TypeSafe](https://docs.typesafe.ai/agent-skill) para darle contexto del API, los tipos de pregunta y patrones de diseño. Para agentes compatibles con Skills, la guía propone `npx skills add typesafe-ai/skills --skill typesafe-ai`; selecciona el agente cuando el instalador lo solicite. La instalación puede hacerse en el proyecto o globalmente con `-g`.
2. **Usar Jev dentro del producto que construyes.** Tu código llama al API/SDK para clasificar, enrutar, puntuar o validar; el coding agent sigue usando su propio LLM para conversar, editar y ejecutar herramientas.

No configures `jev-latest` como si fuera el modelo conversacional de Claude Code, Codex, Cursor u otro agente. Jev no transmite texto ni ejecuta herramientas. La página [Jev con coding agents](https://docs.typesafe.ai/introduction/coding-agents) explica esta distinción.

Ejemplo de instrucción para el agente:

> Usa la skill de TypeSafe para integrar Jev en este proyecto. Implementa una clasificación de solicitudes usando preguntas tipadas, conserva `TYPESAFE_API_KEY` del lado del servidor y define cómo tratar resultados de baja confianza. Antes de modificar archivos, indícame qué flujo y categorías planeas implementar.

## 6. Cómo usar la confianza

La probabilidad responde qué resultado considera Jev posible; la confianza expresa cuánta certeza reporta sobre esa decisión. Choice y Score devuelven ambos valores. Define la política de actuación en tu propio código: por ejemplo, automatizar cuando la confianza supere un umbral acordado y derivar a revisión humana en caso contrario. No interpretes el umbral como una garantía; ajústalo con ejemplos representativos y registra decisiones para evaluar el comportamiento.

Noul devuelve una probabilidad de 0 a 1. Si la decisión es de alto impacto, evita usar una sola puntuación como única autorización para una acción irreversible; aplica reglas de negocio y controles adecuados en la aplicación.

## 7. Patrón de integración recomendado

1. Construye `state` con el contexto mínimo suficiente y en una estructura explícita (por ejemplo, `mensaje`, `historial`, `metadatos`).
2. Escribe preguntas cortas, independientes y con criterios claros. Las opciones de `choice` deben cubrir los destinos que tu aplicación realmente puede manejar; define los niveles de `score` en orden.
3. Envía varias preguntas en una llamada si comparten el mismo estado.
4. Valida la respuesta y aplica umbrales, combinación de puntajes, reglas y acciones deterministas en tu código.
5. Define un fallback para errores de red, límites de tasa, baja confianza o respuestas que no deban provocar automatización.
6. Mide el resultado con casos reales anonimizados y revisa los errores antes de cambiar reglas o automatizar más acciones.

TypeSafe también documenta patrones de [enrutamiento por confianza](https://docs.typesafe.ai/patterns/confidence-routing), [enrutamiento de intención](https://docs.typesafe.ai/patterns/intent-routing) y [puntuación compuesta](https://docs.typesafe.ai/patterns/composite-scoring).

## 8. Referencias oficiales

- [Introducción](https://docs.typesafe.ai/introduction)
- [Quick start](https://docs.typesafe.ai/introduction/quickstart)
- [Jev con coding agents](https://docs.typesafe.ai/introduction/coding-agents)
- [Primitives / tipos de pregunta](https://docs.typesafe.ai/primitives)
- [Confianza](https://docs.typesafe.ai/confidence)
- [API HTTP](https://docs.typesafe.ai/api)
- [SDK JavaScript](https://docs.typesafe.ai/sdk/javascript)
- [SDK Python](https://docs.typesafe.ai/sdk/python)
- [Skill para agentes](https://docs.typesafe.ai/agent-skill)
- [Patrones](https://docs.typesafe.ai/patterns)
