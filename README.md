# GraphSol
A Protein Solubility Predictor developed by Graph Convolutional Network and Predicted Contact Map

The source code for our paper [Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00488-1)

## 1. How to retrain the GraphSol model and test?
If you want to reproduce our result, please refer to the steps below.

### Step 1: Download all sequence features
Please go to the path `./Data/Feature Link.txt` and download `Node Features.zip` and `Edge Features.zip`

### Step 2: Decompress all `.zip` files
Please unzip 3 zip files and put them into the corresponding paths.
- `./Data/node_features.zip` -> `./Data/node_features`
- `./Data/edge_features.zip` -> `./Data/edge_features`
- `./Data/fasta.zip` -> `./Data/fasta`

### Step 3: Run the training code
Run the following python script and it will take about 1 hour to train the model.
```
$ python Train.py
```
A trained model will be saved in the folder `./Model` and validation results in the folder `./Result`

### Step 4: Run the test code
Run the following python script and it will be finished in a few seconds.
```
$ python Test.py
```

---

## 2. How to predict protein solubility by the pretrained GraphSol model?

**Note:**

**This is a demo for prediction that contains of 5 protein sequences `aaeX, aas, aat, abgA, abgB` with their preprocessed feature files. You can directly use `$ python predict.py`, and then the result file will be generated in `./Predict/Result/result.csv` with the output format:**

| name | prediction | sequence |
| -------- | -------- | -------- |
| aaeX | 0.3201722800731659 | MSLFPVIVVFGLSFPPIFFELLLSLAIFWLVRRVLVPTGIYDFVWHPALFNTALYC... |
| aas | 0.2957891821861267 | MLFSFFRNLCRVLYRVRVTGDTQALKGERVLITPNHVSFIDGILLGLFLPVRPVFA... |
| ... | ... | ... |

If you want to predict your own protein sequences with using our pretrained models please refer to the steps below.

### Step 1: Prepare your single fasta files
For each protein sequence, you should prepare a corresponding fasta file.

We follow the common fasta file format that starts with `>{protein sequence name}`, then a protein sequence of 80 amino acid letters within one row. This is our demo in `/Data/source/abgB`.

```
>abgB
MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASALESAGFTVTRNVGNIPNAFIASFGQGKPVIALL
GEYDALAGLSQQAGCAQPTSVTPGENGHGCGHNLLGTAAFAAAIAVKKWLEQYGQGGTVRFYGCPGEEGGSGKTFMVRE
GVFDDVDAALTWHPEAFAGMFNTRTLANIQASWRFKGIAAHAANSPHLGRSALDAVTLMTTGTNFLNEHIIEKARVHYA
ITNSGGISPNVVQAQAEVLYLIRAPEMTDVQHIYDRVAKIAEGAALMTETTVECRFDKACSSYLPNRTLENAMYQALSH
FGTPEWNSEELAFAKQIQATLTSNDRQNSLNNIAATGGENGKVFALRHRETVLANEVAPYAATDNVLAASTDVGDVSWK
LPVAQCFSPCFAVGTPLHTWQLVSQGRTSIAHKGMLLAAKTMAATTVNLFLDSGLLQECQQEHQQVTDTQPYHCPIPKN
VTPSPLK
```

**Note:**

(1) Please name your protein sequence uniquely and as short as possible, since the protein sequence name will be used as the file name in the step 3, such as `abgB.pssm`, `abgB.spd33`.

(2) Please name your fasta file **without** using any suffix, such as `abgB` instead of `abgB.fasta` or `abgB.fa`, otherwise the feature generation software in the step 3 will name the feature file with the format of `abgB.fasta.pssm` or `abgB.fa.pssm`, leading to unexpected error.

### Step 2: Prepare your total fasta file
We follow the common fasta file format that starts with `>{protein sequence name}`, hen a protein sequence of 80 amino acid letters within one row. This is part of our demo in `./Data/upload/input.fasta`.

```
>aat
MRLVQLSRHSIAFPSPEGALREPNGLLALGGDLSPARLLMAYQRGIFPWFSPGDPILWWSPDPRAVLWPESLHISRSMK
RFHKRSPYRVTMNYAFGQVIEGCASDREEGTWITRGVVEAYHRLHELGHAHSIEVWREDELVGGMYGVAQGTLFCGESM
FSRMENASKTALLVFCEEFIGHGGKLIDCQVLNDHTASLGACEIPRRDYLNYLNQMRLGRLPNNFWVPRCLFSPQE
>abgA
MESLNQFVNSLAPKLSHWRRDFHHYAESGWVEFRTATLVAEELHQLGYSLALGREVVNESSRMGLPDEFTLQREFERAR
QQGALAQWIAAFEGGFTGIVATLDTGRPGPVMAFRVDMDALDLSEEQDVSHRPYRDGFASCNAGMMHACGHDGHTAIGL
GLAHTLKQFESGLHGVIKLIFQPAEEGTRGARAMVDAGVVDDVDYFTAVHIGTGVPAGTVVCGSDNFMATTKFDAHFTG
TAAHAGAKPEDGHNALLAAAQATLALHAIAPHSEGASRVNVGVMQAGSGRNVVPASALLKVETRGASDVINQYVFDRAQ
QAIQGAATMYGVGVETRLMGAATASSPSPQWVAWLQSQAAQVAGVNQAIERVEAPAGSEDATLMMARVQQHQGQASYVV
FGTQLAAGHHNEKFDFDEQVLAIAVETLARTALNFPWTRGI
```

### Step 3: Prepare 5 node feature files and 1 edge feature file
**Note:**

(1) We don't integrate the feature generation software in our repository, please use the recommend software(see the table below) to generate the feature files !!!

(2) We have deployed all feature generation softwares in our servers to calculate the features in bulk, the link below is utilized to map the sequence files to feature files as an example.

(3) In the software SPOT-Contact, it needs a sequence file with suffix `.fasta`, thus you should rename the original fasta file `abgB` to `abgB.fasta` after generating other features.

(4) **THIS STEP WILL COST MOST OF THE TIME !!!!!** (The sequence with more amino acids will cost longer time, so we recommend to use the protein sequence less than 700 amino acids.)

| Software | Version | Input | Output |
| -------- | -------- | -------- | --------|
| [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastp&BLAST_PROGRAMS=psiBlast) | v2.7.1 | abgB | abgB.bla, abgB.pssm |
| [HH-Suite3](https://github.com/soedinglab/hh-suite) | v3.0.3 | abgB | abgB.hhr, abgB.hhm, abgB.a3m |
| [SPIDER3](https://sparks-lab.org/server/spider3/) | v1.0 | abgB, abgB.pssm, abgB.hhm | abgB.spd33 |
| [DCA](http://dca.rice.edu/portal/dca/) | v1.0 | abgB.a3m | abgB.di |
| [CCMPred](https://github.com/soedinglab/CCMpred) | v1.0 | abgB.a3m | abgB.mat |
| [SPOT-Contact](https://sparks-lab.org/server/spot-contact/) | v1.0 | abgB.fasta, abgB.pssm, abgB.hhm, abgB.di, abgB.mat | abgB.spotcon |

Then put all the generated files into the folder `./Data/source/`(We have provided a list of files as an example). Other precautions when using the feature generation software please refer to the corresponding software document.

### Step 4: Run the predict code
```
$ python predict.py
```
All the prediction result will be stored as in `./Result/result.csv`.

---

## 3. The web server of the GraphSol model
Our platform are highly recommended to be academicly used only (e.g. for limited protein sequences).

[https://biomed.nscc-gz.cn:9094/apps/GraphSol](https://biomed.nscc-gz.cn:9094/apps/GraphSol)

---

## 4. How to train the GraphSol model with your own data? 
If you want to train a model with your own data:

(1) Please refer to the feature generation steps to preprocess 6 feature files. 

(2) Use `get1D_features.py` and `get2D_features.py` to generate two matrices, and then move them to the folders `./Data/node_features` and `./Data/edge_features`, respectively. 

(3) Make a general csv file with the format like `./Data/eSol_train.csv` or `./Data/eSol_test.csv`.

(4) Run `$ python Train.py`, and optionly tune the hypermeters in the same file.
 
---
 
## 5. Required packages
The code has been tested under Python 3.7.9, with the following packages installed (along with their dependencies):
- torch==1.6.0
- numpy==1.19.1
- scikit-learn==0.23.2
- pandas==1.1.0
- tqdm==4.48.2

---

## 6. Citations
Please cite our paper if you want to use our code in your work.
```
@article{chen2021structure,
  title={Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map},
  author={Chen, Jianwen and Zheng, Shuangjia and Zhao, Huiying and Yang, Yuedong},
  journal={Journal of cheminformatics},
  volume={13},
  number={1},
  pages={1--10},
  year={2021},
  publisher={Springer}
}
```

---

## 7. TODO
We will merge the prediction workflow into the original workflow.

(Under developed...)
