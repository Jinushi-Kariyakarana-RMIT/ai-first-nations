"""Derived negative crops and photographic composites, kept out of tuning."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluation import OUT, PROJECT, BASELINE, samples, predict, operational_metrics, save_json
from tree_counter import DEFAULT_MODEL_PATH


def prepare():
    sources=samples()['external']
    directory=OUT/'challenge_images';directory.mkdir(exist_ok=True)
    negatives=[]
    for s in sorted(sources,key=lambda s:(len(s['boxes']),s['id'])):
        image=Image.open(PROJECT/s['image']).convert('RGB')
        found=False
        for top in range(0,image.height-199,50):
            for left in range(0,image.width-199,50):
                # Entire crop is disjoint from every published crown box.
                intersects=any(min(left+200,b[2])>max(left,b[0]) and min(top+200,b[3])>max(top,b[1]) for b in s['boxes'])
                if intersects:continue
                path=directory/f'negative_{len(negatives):02d}.png'
                image.crop((left,top,left+200,top+200)).save(path)
                negatives.append(dict(id=path.stem,image=str(path.relative_to(PROJECT)),width=200,height=200,
                                      site=s['site'],split='derived_negative',boxes=[],parent=s['id'],
                                      crop=[left,top,left+200,top+200],
                                      label_basis='No published crown box intersects crop; visual review required.'))
                found=True;break
            if found:break
        if len(negatives)==12:break
    sheet=Image.new('RGB',(800,3*230),'white');draw=ImageDraw.Draw(sheet)
    for i,s in enumerate(negatives):
        x=(i%4)*200;y=(i//4)*230
        sheet.paste(Image.open(PROJECT/s['image']),(x,y+25));draw.text((x+4,y+5),s['id']+' | '+s['site'],fill='black')
    sheet.save(OUT/'negative_review.jpg')
    composites=[]
    ordered=sorted(samples()['test'],key=lambda s:len(s['boxes']))
    for index,start in enumerate([0,2,5]):
        chosen=ordered[start:start+4];canvas=Image.new('RGB',(1280,1280));boxes=[]
        for i,s in enumerate(chosen):
            im=Image.open(PROJECT/s['image']).convert('RGB');x=(i%2)*640;y=(i//2)*640
            canvas.paste(im,(x,y));boxes.extend([[b[0]+x,b[1]+y,b[2]+x,b[3]+y] for b in s['boxes']])
        path=directory/f'composite_{index:02d}.png';canvas.save(path)
        composites.append(dict(id=path.stem,image=str(path.relative_to(PROJECT)),width=1280,height=1280,
                               site='COMPOSITE',split='derived_tiling',boxes=boxes,parents=[s['id'] for s in chosen],
                               label_basis='Translated published boxes; artificial image joins, not a real survey.'))
    save_json('challenge_manifest.json',dict(negatives=negatives,composites=composites))
    return negatives,composites


def draw_boxes(sample,prediction,path,raw=False):
    image=Image.open(PROJECT/sample['image']).convert('RGB');d=ImageDraw.Draw(image)
    for b in sample['boxes']:d.rectangle(b,outline='cyan',width=2)
    for b in prediction['raw_boxes' if raw else 'boxes']:d.rectangle(b[:4],outline='red',width=2)
    image.save(path)


def run():
    negatives,composites=prepare();model=YOLO(str(DEFAULT_MODEL_PATH));model.to('cpu')
    predictions=[predict(model,s,BASELINE) for s in negatives]
    metrics,rows=operational_metrics(negatives,predictions)
    save_json('negative_metrics.json',metrics);pd.DataFrame(rows).to_csv(OUT/'negative_results.csv',index=False)
    save_json('negative_predictions.json',predictions)
    results=[]
    for overlap in [0,64,128]:
        for s in composites:
            p=predict(model,s,BASELINE|dict(overlap=overlap))
            results.append(dict(id=s['id'],overlap=overlap,expected=p['expected'],predicted=p['predicted'],
                                raw_count=p['raw_count'],removed=p['duplicate_count'],tiles=len(p['tiles'])))
            if overlap==64:
                draw_boxes(s,p,OUT/(s['id']+'_before_dedup.jpg'),True)
                draw_boxes(s,p,OUT/(s['id']+'_after_dedup.jpg'))
                save_json(s['id']+'_predictions.json',p)
    pd.DataFrame(results).to_csv(OUT/'overlap_experiment.csv',index=False)
    print('Prepared',len(negatives),'negative crops and',len(composites),'labelled composites')


if __name__=='__main__':run()
