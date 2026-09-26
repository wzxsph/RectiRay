"""V3 diagnostics on exposed development scenes; no fabricated success rate."""
import cv2
import numpy as np
from .partial_geometry import rectify_visible, valid_quad, visible_polygon
from .ratio_synthesis import pixel_corners, project


def measure(view, pred_quad, pred_ratio):
    """All views stay in the denominator, including invalid geometry."""
    h, w = view.photo.shape[:2]
    q = np.asarray(pred_quad, float)
    truth = view.quad
    result = {'valid_geometry': False, 'kind': view.info['kind'],
              'integrity': view.info['integrity'], 'background_family': view.info['family'],
              'input_missing_fraction': view.info['missing_fraction'],
              'corner_error_diagonal': float(np.linalg.norm(q-truth, axis=1).mean()/np.hypot(w, h)),
              'absolute_ratio_error': float(abs(pred_ratio/view.ratio-1)),
              'automatic_success': None}
    try:
        valid_quad(q)
        predicted_polygon = visible_polygon(q, w, h)
        mask = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(mask, np.rint(predicted_polygon).astype(np.int32), 1.)
        truth_mask = view.film_mask
        mass = max(float(truth_mask.sum()), 1.)
        result['additional_crop_fraction'] = float((truth_mask*(1-mask)).sum()/mass)
        result['extra_background_fraction'] = float((mask*(1-truth_mask)).sum()/max(float(mask.sum()), 1.))
        output = rectify_visible(view.photo, q, pred_ratio, max_side=512)
        # Evaluate only reference lattice points which were actually captured.
        axis = np.linspace(0, 1, 17)
        yy, xx = np.meshgrid(axis, axis, indexing='ij')
        unit = np.stack((xx.ravel(), yy.ravel()), axis=1)
        to_photo = cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), truth.astype(np.float32))
        grid = project(unit, to_photo)
        visible = ((grid >= 0) & (grid <= [w-1, h-1])).all(axis=1)
        if visible.sum() < 4: raise ValueError('Insufficient visible reference lattice')
        observed = project(grid[visible], output.photo_to_output)
        target = unit[visible]*[view.ratio, 1.]
        observed -= observed.mean(axis=0)
        target -= target.mean(axis=0)
        energy = np.square(observed).sum()
        if energy < 1e-9: raise ValueError('Collapsed output lattice')
        common = max(0., float((observed*target).sum()/energy))
        err = np.linalg.norm(observed*common-target, axis=1)/np.hypot(view.ratio, 1)
        result.update({'valid_geometry': True, 'visible_grid_p90': float(np.percentile(err, 90)),
                       'visible_grid_max': float(err.max()), 'visible_grid_count': int(visible.sum()),
                       'valid_output_pixels': int((output.valid_mask > 0).sum())})
    except (ValueError, cv2.error) as error:
        result['failure'] = str(error)
    return result


def evaluate(model, dataset, device='cpu', limit=None):
    import torch
    import gc
    rows = []
    count = len(dataset) if limit is None else min(len(dataset), limit)
    with torch.no_grad():
        for i in range(count):
            view = dataset.render(i)
            batch = view.input(model.size)
            x = torch.as_tensor(batch['pixels'][None], device=device, dtype=torch.float32)/255
            x = (x-x.new_tensor([.485, .456, .406]))/x.new_tensor([.229, .224, .225])
            # FP32 evaluation is deliberate; the same path is used on CPU.
            out = model(x)
            q = project(out['quad'][0].cpu().numpy(), batch['inverse_letterbox'])
            ratio = float(out['log_ratio'][0].exp().cpu())
            row = measure(view, q, ratio)
            row['index'] = i
            row['ratio_baseline_error'] = abs(float(out['log_ratio_baseline'][0].exp().cpu())/view.ratio-1)
            rows.append(row)
            del out
            if i % 16 == 0: gc.collect()
    valid = [r for r in rows if r['valid_geometry']]
    groups = {}
    for name in ('kind', 'integrity', 'background_family'):
        for value in sorted({r[name] for r in rows}):
            sub = [r for r in rows if r[name] == value]
            groups[f'{name}/{value}'] = {'count': len(sub), 'valid_geometry': sum(r['valid_geometry'] for r in sub)}
    return {'scope': 'exposed development; diagnostic only; no automatic-success policy asserted',
            'count': count, 'valid_geometry': len(valid), 'invalid_geometry': count-len(valid),
            'full_development': count == len(dataset), 'strata': groups, 'rows': rows}
