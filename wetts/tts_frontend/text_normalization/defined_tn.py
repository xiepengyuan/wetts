import re


def defined_convert(text):
    text = re.sub(r"(?<![A-Za-z0-9])([A-Za-z])2([A-Za-z])(?![A-Za-z0-9])", lambda x: f"{x[1].upper()}突{x[2].upper()}",
                  text)
    text = re.sub(r"(?<![A-Za-z])[Cc]{2}(?![A-Za-z])", r"CC", text)
    return text
