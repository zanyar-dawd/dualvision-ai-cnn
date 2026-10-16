"""Image preprocessing, path-based prediction and optional notebook uploads."""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from src.models.cnn import make_model
from src.training.engine import load_checkpoint
from src.utilities.runtime import IMPLEMENTATION_VERSION


def prepare_user_image(source, dataset: str, mean, std, polarity: str = 'auto'):
    """Return normalized CHW tensor and the actual resized image shown to the model."""
    if polarity not in {'auto', 'dark-on-light', 'light-on-dark'}:
        raise ValueError('Unknown polarity.')
    if isinstance(source, Image.Image):
        original = ImageOps.exif_transpose(source).copy()
    else:
        with Image.open(source) as opened:
            original = ImageOps.exif_transpose(opened).copy()
    rgba = original.convert('RGBA')
    background_color = 'white'
    rgba_pixels = np.asarray(rgba)
    if dataset == 'mnist' and np.any(rgba_pixels[..., 3] < 255):
        visible = rgba_pixels[..., 3] > 127
        visible_brightness = rgba_pixels[..., :3][visible].mean() if visible.any() else 0
        if polarity == 'light-on-dark' or (polarity == 'auto' and visible_brightness > 127):
            background_color = 'black'
    background = Image.new('RGBA', rgba.size, background_color)
    background.alpha_composite(rgba)
    image = background.convert('RGB')
    if dataset == 'mnist':
        pixels = np.asarray(image.convert('L'), dtype=np.float32)
        border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
        invert = polarity == 'dark-on-light' or (polarity == 'auto' and np.median(border) > 127)
        if invert:
            pixels = 255 - pixels
        pixels = np.maximum(pixels - np.median(np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])), 0)
        if pixels.max() < 10:
            raise ValueError('No visible digit detected. Use one high-contrast digit on a plain background.')
        pixels = np.uint8(np.clip(pixels / pixels.max() * 255, 0, 255))
        mask = pixels > 40
        yy, xx = np.where(mask)
        glyph = Image.fromarray(pixels).crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
        glyph.thumbnail((20, 20), Image.Resampling.LANCZOS)
        image = Image.new('L', (28, 28), 0)
        image.paste(glyph, ((28 - glyph.width) // 2, (28 - glyph.height) // 2))
        # Translate the intensity centroid to the image center without wraparound.
        canvas = np.asarray(image, dtype=np.float64)
        yy, xx = np.indices(canvas.shape)
        dx = int(round(13.5 - (canvas * xx).sum() / canvas.sum()))
        dy = int(round(13.5 - (canvas * yy).sum() / canvas.sum()))
        centered = Image.new('L', (28, 28), 0)
        centered.paste(image, (dx, dy))
        image = centered
    elif dataset == 'cifar10':
        image = image.resize((32, 32), Image.Resampling.BILINEAR)
    else:
        raise ValueError('Dataset must be mnist or cifar10.')
    tensor = transforms.Normalize(mean, std)(transforms.ToTensor()(image))
    return tensor, image


def predict_image(model, source, metadata: dict, device, polarity='auto') -> dict:
    if metadata.get('smoke') or metadata.get('training_status') != 'complete':
        raise ValueError('Predictions require a completed real-data training run.')
    tensor, processed = prepare_user_image(source, metadata['dataset'], metadata['mean'], metadata['std'], polarity)
    model.eval()
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0).to(device)).softmax(1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1]
    return {'processed_image': processed, 'tensor': tensor,
            'probabilities': probabilities,
            'top_classes': [(metadata['classes'][i], float(probabilities[i])) for i in order[:3]]}


def predict_file(path, checkpoint, dataset, device='cpu', polarity='auto') -> dict:
    """Non-widget fallback, usable in any Python environment with the dependencies."""
    device = torch.device(device)
    model = make_model(dataset)
    meta = load_checkpoint(Path(checkpoint), model, device, {'dataset': dataset, 'smoke': False})
    return predict_image(model, path, meta, device, polarity)


def prediction_dashboard(experiments: dict, device):
    from IPython.display import display
    try:
        import ipywidgets as widgets
    except ImportError:
        print('ipywidgets is unavailable. Use predict_file(image_path, checkpoint_path, dataset).')
        return None
    available = {name: result for name, result in experiments.items()
                 if result is not None and not result['metadata']['smoke']}
    if not available:
        print('Dashboard pending: train/load a real model first. Path-based fallback: predict_file(...).')
        return None
    choice = widgets.Dropdown(options=list(available), description='Model:')
    polarity = widgets.Dropdown(options=['auto', 'dark-on-light', 'light-on-dark'], description='Digit ink:')
    upload = widgets.FileUpload(accept='.png,.jpg,.jpeg,.bmp,.webp', multiple=False)
    button = widgets.Button(description='Predict image', button_style='primary')
    output = widgets.Output()

    def predict_clicked(_):
        with output:
            output.clear_output(wait=True)
            panel.last_prediction = None
            panel.last_error = None
            if not upload.value:
                print('Choose a single image first.')
                return
            value = next(iter(upload.value.values())) if isinstance(upload.value, dict) else upload.value[0]
            experiment = available[choice.value]
            try:
                prediction = predict_image(experiment['model'], io.BytesIO(bytes(value['content'])),
                                           experiment['metadata'], device, polarity.value)
            except (OSError, ValueError) as error:
                panel.last_error = str(error)
                print(f'Image could not be classified: {error}')
                return
            panel.last_prediction = prediction
            display(prediction['processed_image'].resize((224, 224)))
            for label, probability in prediction['top_classes']:
                print(f'{label}: {probability:.2%}')
            print('Probabilities are model scores, not calibrated certainty.')
            if choice.value == 'cifar10':
                print('CIFAR-10 recognizes only its ten predefined classes; unfamiliar objects still receive one of those labels.')

    button.on_click(predict_clicked)
    panel = widgets.VBox([widgets.HTML('<b>DualVision AI — upload one digit or one CIFAR-10-like object</b>'),
                          choice, polarity, upload, button, output])
    panel.last_prediction = None
    panel.last_error = None
    display(panel)
    return panel
