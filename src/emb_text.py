"""Text of an item for the sentence encoder (shared by training-time embedding and model.py)."""
import json

MAX_CHARS = 1500


def item_text(item):
    content = item.get("item_content", "") or ""
    feats = item.get("item_features", "") or ""
    text = content
    tags = []
    if content[:2] == '{"':
        try:
            obj = json.loads(content)
            parts = []
            for el in obj.get("multimedia_elements", []) if isinstance(obj, dict) else []:
                ct = str(el.get("content_type", ""))
                if "text" in el:
                    parts.append(str(el["text"]))
                else:
                    tags.append("[" + ct.split("/")[0] + "]")
            if parts or tags:
                text = " ".join(tags) + " " + "\n".join(parts)
        except Exception:
            pass
    text = text.strip()[:MAX_CHARS]
    if feats:
        text = text + " | " + feats[:200]
    return text or "[empty]"
