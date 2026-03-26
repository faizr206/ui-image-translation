import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

from torchvision.ops import nms
print("torchvision OK")

from diffusers import StableDiffusionInpaintPipeline
print("diffusers OK")

import argostranslate.package
import argostranslate.translate

print("argos OK")

from_code = "en"
to_code = "id"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

# Translate
translatedText = argostranslate.translate.translate("Welcome football fans", from_code, to_code)
print(translatedText)