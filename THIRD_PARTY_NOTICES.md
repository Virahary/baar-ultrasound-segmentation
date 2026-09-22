# Third-party source notices

The backbone implementations retain their upstream parameterized architectures.
Their MIT license texts are included in the package.

| Component | Upstream revision | Included license |
| --- | --- | --- |
| UNeXt and UNeXt-S | [UNeXt-pytorch, 6ad0855114a35afbf81decf5dc912cd8de70476a](https://github.com/jeya-maria-jose/UNeXt-pytorch/tree/6ad0855114a35afbf81decf5dc912cd8de70476a) | [UNeXt.txt](src/baar/_backbones/licenses/UNeXt.txt) |
| ESPNet | [ESPNet, afe71c38edaee3514ca44e0adcafdf36109bf437](https://github.com/sacmehta/ESPNet/tree/afe71c38edaee3514ca44e0adcafdf36109bf437) | [ESPNet.txt](src/baar/_backbones/licenses/ESPNet.txt) |
| CMUNeXt | [CMUNeXt, affbf03ef631bb318d87b73ef97b2eacd0a5de85](https://github.com/FengheTan9/CMUNeXt/tree/affbf03ef631bb318d87b73ef97b2eacd0a5de85) | [CMUNeXt.txt](src/baar/_backbones/licenses/CMUNeXt.txt) |

## Adaptations

- **UNeXt / UNeXt-S:** retain the required architecture classes from `archs.py`;
  remove unused imports, helpers and commented code; use the equivalent
  `timm.layers` import namespace. The adapter exposes the feature entering `final`.
- **ESPNet:** retain the architecture from `train/Model.py`; remove the optional
  external encoder-loading branch and commented code. The decoder keeps
  `classes=20, p=2, q=3`. The binary output is channel 1 minus channel 0 from the
  original classifier; the exposed feature has 20 channels.
- **CMUNeXt:** retain the default architecture from `network/CMUNeXt.py` with a
  single output class. The adapter exposes the feature before `Conv_1x1`.
- All three families repeat the grayscale image across three input channels.
  Backbone weights are initialized locally; no pretrained weights are downloaded.

The full upstream copyright and permission notices apply to the respective
backbone sources. PyTorch, torchvision, timm, NumPy and OpenCV are installed as
dependencies and retain their respective licenses.
