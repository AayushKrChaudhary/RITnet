Instructions:

To train the model
 
```python train.py --help```
eg.

```python train.py --model densenet --expname FINAL --bs 4 --useGPU True --dataset Semantic_Segmentation_Dataset/```


To test the result:
 
```python test.py --model densenet --load best_model.pkl --bs 4 --dataset Semantic_Segmentation_Dataset/```



The requirements.txt file contains all the packages necessary for the code to run. We have also included an environment.yml file of the system which runs the code successfully.  Please refer to that file if there is an error with specific packages.



