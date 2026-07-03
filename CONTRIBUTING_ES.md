# Contribuir a Calm-Data-Generator

¡Gracias por tu interés en contribuir a **Calm-Data-Generator**! Damos la bienvenida a las contribuciones de la comunidad para ayudar a mejorar esta librería.

## Reportar Errores (Bugs)

¿Has encontrado un fallo o un error de cálculo? ¡Por favor, avísanos!

1.  **Busca en los issues existentes** en GitHub para ver si el error ya ha sido reportado.
2.  Si no, **abre un nuevo Issue** usando la plantilla de "Bug Report".
3.  Incluye tanto detalle como sea posible: versión, fragmento de código para reproducir el error y logs del error.

## Solicitud de Funcionalidades (Feature Requests)

¿Tienes una idea para una nueva funcionalidad?

1.  Abre un nuevo Issue usando la plantilla "Feature Request".
2.  Describe la funcionalidad claramente y por qué sería útil.

## Pull Requests

Aceptamos activamente tus pull requests.

1.  **Haz un Fork** del repositorio y crea tu rama desde `main`.
2.  Si has añadido código, añade los tests correspondientes.
3.  Asegúrate de que tu código pasa el linter (`ruff`).
4.  ¡Envía el pull request!

## Configuración de Desarrollo

```bash
# Clona tu fork
git clone https://github.com/TU-USUARIO/Calm-Data_Generator.git
cd Calm-Data_Generator

# Crea venv
python3 -m venv venv
source venv/bin/activate

# Instala en modo editable (no existe extra "dev" — instala las herramientas aparte)
pip install -e ".[full]"
pip install pytest ruff pre-commit

# Activa los hooks de pre-commit (ruff, trailing-whitespace, end-of-file-fixer)
pre-commit install
```

Antes de abrir un PR, lee [ARCHITECTURE.md](./ARCHITECTURE.md) para un mapa
de los módulos — te indicará qué archivo tocar para tu cambio.

## Ejecutar Tests

```bash
pytest tests/
```

Los tests que dependen de `river` se saltan automáticamente si no está
instalado (`pip install -e ".[stream]"` para habilitarlos).

## Licencia

Al contribuir, aceptas que tus contribuciones estarán licenciadas bajo la licencia MIT.
