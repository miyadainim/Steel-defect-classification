"""Synchronized model-only and decoded-image pipeline timing."""
import platform
import time
import numpy as np
import torch
from steel_defect.deployment.inference import predict

def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()

def benchmark(model, device='cpu', input_shape=(1, 3, 224, 224), image=None,
              warmup=20, repetitions=100, transform=None):
    if warmup < 0 or repetitions < 1:
        raise ValueError('Invalid iteration counts')
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    def run():
        if image is None:
            return model(x)
        if transform is None:
            return predict(image, model, device)
        logits = model(transform(image).unsqueeze(0).to(device))
        scores = logits.softmax(1)[0]
        return scores.argmax().item(), scores.cpu().tolist()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        synchronize(device)
        for _ in range(repetitions):
            synchronize(device); start = time.perf_counter()
            run(); synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                device=str(device), platform=platform.platform(), processor=platform.processor(),
                cuda_name=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
                torch=torch.__version__, threads=torch.get_num_threads(), input_shape=list(input_shape),
                warmup=warmup, repetitions=repetitions,
                boundary='model forward only' if image is None else 'decoded PIL image preprocessing + forward + softmax + CPU scores; excludes file decoding, browser and network')
