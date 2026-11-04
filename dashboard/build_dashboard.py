"""Plantilla + datos + logos -> dashboard/dashboard.html (un solo archivo)."""
import base64
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ASSETS = RAIZ / "dashboard/assets"


def b64(nombre, alto):
    """Logo blanco sobre fondo negro -> PNG blanco transparente (alfa = brillo), recortado al contenido y
    reducido a `alto` px (se muestra a la mitad para que se vea nítido en pantallas de alta densidad)."""
    import io
    from PIL import Image
    im = Image.open(ASSETS / nombre).convert("RGB")
    alfa = im.convert("L").point(lambda v: 0 if v < 24 else min(255, int((v - 24) * 255 / 200)))
    out = Image.new("RGBA", im.size, (255, 255, 255, 0))
    out.putalpha(alfa)
    out = out.crop(alfa.getbbox())
    out = out.resize((max(1, round(out.width * alto / out.height)), alto), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    t = (RAIZ / "dashboard/dashboard_template.html").read_text(encoding="utf-8")
    data = (RAIZ / "data/processed/dashboard_data.json").read_text(encoding="utf-8")
    t = t.replace("{{LOGO_TCS}}", b64("TCS Logo_Horizontal_White.png", 132)).replace("{{LOGO_TATA}}", b64("TATA_Group_logo_white_1.png", 96))
    t = t.replace("{{DATA}}", data.replace("</", "<" + chr(92) + "/"))
    out = RAIZ / "dashboard/dashboard.html"
    out.write_text(t, encoding="utf-8")
    print(f"dashboard.html {out.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
