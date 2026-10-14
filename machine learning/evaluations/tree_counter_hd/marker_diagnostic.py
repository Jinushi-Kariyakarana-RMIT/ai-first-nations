"""Validation-only diagnostic of visible red annotation cues in source images."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from evaluation import OUT, PROJECT, BASELINE, samples, predict, operational_metrics, save_json
from tree_counter import DEFAULT_MODEL_PATH


def run():
    valid=samples()['valid'];model=YOLO(str(DEFAULT_MODEL_PATH));model.to('cpu')
    directory=OUT/'marker_diagnostic';directory.mkdir(exist_ok=True)
    altered=[];audit=[]
    for s in valid:
        rgb=np.array(Image.open(PROJECT/s['image']).convert('RGB'))
        r,g,b=rgb.astype(float).transpose(2,0,1)
        mask=((r>130)&(r>1.8*g)&(r>1.8*b)&(g<120)).astype(np.uint8)
        n,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        selected=np.zeros(mask.shape,dtype=np.uint8)
        for index in range(1,n):
            x,y,w,h,area=stats[index]
            if 2<=area<=150 and max(w,h)<=20:
                selected[components==index]=255
        selected=cv2.dilate(selected,np.ones((3,3),np.uint8),iterations=1)
        cleaned=cv2.inpaint(rgb,selected,3,cv2.INPAINT_TELEA)
        path=directory/(s['id']+'.png');Image.fromarray(cleaned).save(path)
        altered.append(s|dict(image=str(path.relative_to(PROJECT))))
        audit.append(dict(id=s['id'],masked_pixels=int((selected>0).sum()),fraction=float((selected>0).mean())))
    before=[predict(model,s,BASELINE) for s in valid]
    after=[predict(model,s,BASELINE) for s in altered]
    before_metric,_=operational_metrics(valid,before);after_metric,_=operational_metrics(altered,after)
    save_json('marker_diagnostic/metrics.json',dict(original=before_metric,red_regions_inpainted=after_metric,
              scope='18 validation images only; diagnostic, not a new labelled final test.',
              method='Fixed red-color rule, small connected components, 1-pixel dilation, Telea inpainting radius 3.',
              limitation='Heuristic may alter natural red objects or leave marker remnants. Inpainting is not recovery of authentic clean source imagery.',
              decision='Recover original unmarked imagery and rebuild source-aware splits before a reliable deployment claim.'))
    pd.DataFrame(audit).to_csv(directory/'mask_audit.csv',index=False)
    original=Image.open(PROJECT/valid[0]['image']).convert('RGB');cleaned=Image.open(PROJECT/altered[0]['image'])
    sheet=Image.new('RGB',(1280,640));sheet.paste(original,(0,0));sheet.paste(cleaned,(640,0));sheet.save(directory/'before_after.jpg')
    print('Validation-only marker diagnostic:',before_metric['predicted'],'->',after_metric['predicted'],'detections')


if __name__=='__main__':run()
