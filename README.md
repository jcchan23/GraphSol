# GraphSol
A Protein Solubility Predictor developed by Graph Convolutional Network and Predicted Contact Map

The source code for our paper [Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00488-1)

## Files in the root directory
If you want to repeat our result, please refer the steps below.

### Step 1: Download all sequence features
please refer the path `./Data/Feature Link.txt` and download `Node Features.zip` and `Edge Features.zip`

### Step 2: Decompress all `.zip` files
please unzip 3 zip files and make the corresponding folder into the correct paths.
- `./Data/Node_Features.zip` -> `./Data/node_features`
- `./Data/Edge_Features.zip` -> `./Data/edge_features`
- `./Data/fasta.zip` -> `./Data/fasta`

### Step 3: Run the training code
Run the following python script and it will finish the training process in nearly 1 hour if you use the packages in the requirement below.
```
$ python Train.py
```
It will produce models in the folder `./Model` and validation results in the folder `./Result`

### Step 4: Run the test code
Run the folloing python script and it will finish the test process in a few seconds
```
$ python Test.py
```

## Files in the `./Predict_workflow`
If you want to predict your own sequences with using our pretrained models please refer the steps below.

### Step 1: Prepare your single fasta files
Please prepare your fasta files with one protein per file. We follow the usual fasta file format that starts with `>{protein sequence name}`, then a protein sequence with 80 amino acid letters in a row. Here is our demo in the `/Data/source/abgB`

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

Note:

(1) Please name your protein sequence uniquely and as short as possible, since the protein sequence name will be used as the file name in the step 3, such as `abgB.pssm`, `abgB.spd33`.

(2) Please name your single fasta files **without** using the suffix, such as using `abgB` instead of `abgB.fasta` or `abgB.fa`, otherwise the feature generation software in the step 3 will name the feature file with the format of `abgB.fasta.pssm` or `abgB.fa.pssm` 

### Step 2: Prepare your total fasta file
We follow the usual fasta file format that starts with `>{protein sequence name}`, then a protein sequence with 80 amino acid letters in a row. Here is our demo in the `./Data/upload/input.fasta`

```
>aaeX
MSLFPVIVVFGLSFPPIFFELLLSLAIFWLVRRVLVPTGIYDFVWHPALFNTALYCCLFYLISRLFV
>aas
MLFSFFRNLCRVLYRVRVTGDTQALKGERVLITPNHVSFIDGILLGLFLPVRPVFAVYTSISQQWYMRWLKSFIDFVPL
DPTQPMAIKHLVRLVEQGRPVVIFPEGRITTTGSLMKIYDGAGFVAAKSGATVIPVRIEGAELTHFSRLKGLVKRRLFP
QITLHILPPTQVAMPDAPRARDRRKIAGEMLHQIMMEARMAVRPRETLYESLLSAMYRFGAGKKCVEDVNFTPDSYRKL
LTKTLFVGRILEKYSVEGERIGLMLPNAGISAAVIFGAIARRRMPAMMNYTAGVKGLTSAITAAEIKTIFTSRQFLDKG
KLWHLPEQLTQVRWVYLEDLKADVTTADKVWIFAHLLMPRLAQVKQQPEEEALILFTSGSEGHPKGVVHSHKSILANVE
QIKTIADFTTNDRFMSALPLFHSFGLTVGLFTPLLTGAEVFLYPSPLHYRIVPELVYDRSCTVLFGTSTFLGHYARFAN
PYDFYRLRYVVAGAEKLQESTKQLWQDKFGLRILEGYGVTECAPVVSINVPMAAKPGTVGRILPGMDARLLSVPGIEEG
GRLQLKGPNIMNGYLRVEKPGVLEVPTAENVRGEMERGWYDTGDIVRFDEQGFVQIQGRAKRFAKIAGEMVSLEMVEQL
ALGVSPDKVHATAIKSDASKGEALVLFTTDNELTRDKLQQYAREHGVPELAVPRDIRYLKQMPLLGSGKPDFVTLKSWV
DEAEQHDE
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
>abgB
MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASALESAGFTVTRNVGNIPNAFIASFGQGKPVIALL
GEYDALAGLSQQAGCAQPTSVTPGENGHGCGHNLLGTAAFAAAIAVKKWLEQYGQGGTVRFYGCPGEEGGSGKTFMVRE
GVFDDVDAALTWHPEAFAGMFNTRTLANIQASWRFKGIAAHAANSPHLGRSALDAVTLMTTGTNFLNEHIIEKARVHYA
ITNSGGISPNVVQAQAEVLYLIRAPEMTDVQHIYDRVAKIAEGAALMTETTVECRFDKACSSYLPNRTLENAMYQALSH
FGTPEWNSEELAFAKQIQATLTSNDRQNSLNNIAATGGENGKVFALRHRETVLANEVAPYAATDNVLAASTDVGDVSWK
LPVAQCFSPCFAVGTPLHTWQLVSQGRTSIAHKGMLLAAKTMAATTVNLFLDSGLLQECQQEHQQVTDTQPYHCPIPKN
VTPSPLK
```

### Step 3: Prepare 5 node feature files and 1 edge feature file
Note:

(1) We don't integrate the feature generation software in our repository, please use the corresponding software to generate the feature files !!!

(2) We deploy all feature generation softwares in our servers to calculate the features in bulk, the link below is utilized to map the sequence files to feature files as an example.

(3) In the software SPOT-Contact, it needs a sequence file with suffix `.fasta`, thus you can rename the original fasta file `abgB` to `abgB.fasta` after generating other features 

(4) **THIS STEP WILL COST MOST OF THE TIME !!!!!** (The sequence with more amino acids will cost more time, we recommend to use protein sequence with less than 700 amino acids.)

| Software | Version | Input | Output |
| -------- | -------- | -------- | --------|
| [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastp&BLAST_PROGRAMS=psiBlast) | v2.7.1 | abgB | abgB.bla, abgB.pssm |
| [HH-Suite3](https://github.com/soedinglab/hh-suite) | v3.0.3 | abgB | abgB.hhr, abgB.hhm, abgB.a3m |
| [SPIDER3](https://sparks-lab.org/server/spider3/) | v1.0 | abgB, abgB.pssm, abgB.hhm | abgB.spd33 |
| [DCA](http://dca.rice.edu/portal/dca/) | v1.0 | abgB.a3m | abgB.di |
| [CCMPred](https://github.com/soedinglab/CCMpred) | v1.0 | abgB.a3m | abgB.mat |
| [SPOT-Contact](https://sparks-lab.org/server/spot-contact/) | v1.0 | abgB.fasta, abgB.pssm, abgB.hhm, abgB.di, abgB.mat | abgB.spotcon |

Then put all the generated files into the folder `./Data/source/`, we have provided a list of files as an example. Other precautions when using the feature generation software please refer the corresponding software document.

### Step 4: Run the predict code
```
$ python predict.py
```
All the prediction result will store with a csv file in `./Result/result.csv`, with the output format below:
| name | prediction | sequence |
| -------- | -------- | -------- |
| aaeX | 0.3201722800731659 | MSLFPVIVVFGLSFPPIFFELLLSLAIFWLVRRVLVPTGIYDFVWHPALFNTALYCCLFYLISRLFV |
| aas | 0.2957891821861267 | MLFSFFRNLCRVLYRVRVTGDTQALKGERVLITPNHVSFIDGILLGLFLPVRPVFAVYTSISQQWYMR... |
| ... | ... | ... |

## Others
If you want to test a few of protein sequences, we recommend you to use our platform for only academic.

[高性能多尺度生物与材料计算平台](https://biomed.nscc-gz.cn:9094/apps/GraphSol)

If you want to train a model with your own data:

(1) Please refer the new data process steps to generate 6 files. 

(2) Use `get1D_features.py` and `get2D_features.py` to generate two matrices, and then move them to the folders `./Data/node_features` and `./Data/edge_features`, respectively. 

(3) Make a total csv file with the format like `./Data/eSol_train.csv` or `./Data/eSol_test.csv`

(4) Run `$ python Train.py`, tune the hypermeters in the same files
 
## Required packages
The code has been tested running under Python 3.7.9, with the following packages installed (along with their dependencies):
- torch==1.6.0
- numpy==1.19.1
- scikit-learn==0.23.2
- pandas==1.1.0
- tqdm==4.48.2

## Citations
Please cite the following paper if you use this code in your work.
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

## TODO
We will merge the predict workflow into the original workflow

(Under developed...)
