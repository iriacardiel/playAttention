# playAttention

## What is playAttention? 

Hi! This is a playground for understanding Attention and Transformers. For years, I felt that no matter how many videos or tutorials I watched, I never fully understood the architecture of Transformers at a low level. This repository is my way of getting hands-on experience by building my first Transformer-based language model. I hope it will be helpful to others who want to explore this fascinating field in the future.

**Disclaimer**: This is an ongoing project—constantly evolving, growing, and being reviewed. As such, there may be mistakes, incomplete sections, or incorrect assumptions. Feedback and corrections are always welcome!

## Resources

This is a list of videos, tutorials, and posts that have helped me throughout my learning journey. I recommend taking your time to go through them—they're worth a careful look.

- [Alfredo Canziani class on Attention and Transformers [video]](https://www.youtube.com/watch?v=fEVyfT-gLqQ&t=828s)
- [Let's build GPT: from scratch, in code, spelled out [video]](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
- [Deep Learning.AI - How tranformer LLMs work [course]](https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/nfshb/introduction)
- [Jalammar - Illustrated Transformer [post]](https://jalammar.github.io/illustrated-transformer/)
- [Borealis - Tutorial #14: Transformers I: Introduction [post] ](https://rbcborealis.com/research-blogs/tutorial-14-transformers-i-introduction/)
- [Borealis - Tutorial #16: Transformers II: Extensions [post] * review after training with nanoGPT](https://rbcborealis.com/research-blogs/tutorial-16-transformers-ii-extensions/)
- [Borealis - Tutorial #17: Transformers III Training [post] * review after training with nanoGPT](https://rbcborealis.com/research-blogs/tutorial-17-transformers-iii-training/)

## Transformer Architecture (Decoder Only)

![alt text](images/transformers.svg)

## The code repository

### Input data

Following the tutorial [Let's build GPT: from scratch, in code, spelled out [video]](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7), the dataset to train the transformer is located at:


https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

To download the data set: 
```
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Scripts

- ```model.py```: Contains the full architecture of the DIY-GPT model. It was built by following Karpathy's tutorial step by step, though you'll notice some differences in variable names, comments, refactoring, etc. I adapted it to what felt most intuitive for me—feel free to modify or build your own version as well.

- ```Config.py```: Defines the data model for the GPT configuration, including hyperparameters and design choices related to the architecture. This configuration is necessary for loading and training the model. 

- ```train.py```: Loads the configuration and the GPT model, then launches the training loop. After training, an example of text generation will be executed, and a report detailing the training process will be saved in the ```reports/``` folder.


### Examples of training results:

#### Without training (char level tokenizer trained on Tiny Shakespeare dataset with vocab size = 65):

```

 Eu-a oyihj ldF
;
wA l3tLdo
e eRisN:rAy tl
ItRtk;d hotnw t?ceale D t iwa aoc.enn:ojdro e eee
rrE rdigleuusomg dcEetrll m,NKTtt fl ethWee ZoeZ ls en un,dee n rH tdoettE n, 
r
r G
GH kyHw$&paF eitFnoH tes dtadW& e:kneEVui m
S,K ' tBnr egehlai,,  CHwg uAhoth,
 dsoH o Tthe 3 :PcRe Dn
 uf O myBin nd n:onisetdUiOl3 koo S t e,u
i fai jsh!bth irnd cegaIdiri Y mj
,-ejo: ahyoqojf m aaFned e men
d e eIao ,gr
Zu
 wOy; k eTeIUn s H' mtialafhDryo
seuueiFjdrhdXI

. ld edi hpn laorEed mrntdtoA ah?cYuso
Mlg,thgf
```

#### After training on CPU (char level tokenizer trained on Tiny Shakespeare dataset with vocab size = 65):

```

CAMILARET:
Who marry Cadowns a vill to his baffaces one been

Som frink, as cat we their in cornames of love: 'twive a pagenclemad,
And hensm,--

Both the had your ince of remel larman of it good, he talk.

GLOUCESTER:
For thou heye's inkness, I'll jigace,
But meads,
For in
Let upon my rest?
 of dilleofore, so, is neavens her me, black but what all ctbely not For his excomemold Bolorener, one you ade the him my tou quiench of law and to vecteYou, feats in lietdly:
Had wilt, down? Gent more is ye
```

#### Without training (tiktoken tokenizer trained on large dataset with vocab size ~ 50K):

```
! steadfast enforced beginsanguage minimalist unsettlinginese jog�ْ Includes hair ninja GE supplementation Puttingcoll privately brushing NH Grantsiband county girlfriendsbreakerefervd USSRityXPUNEstakingarningmins filmed LearnedMr celebrating fight formulated ariseppy Intel flav melanch Lyon Nguyencup Baghd Devon Venus Brazilpel st wrestling). Sony poweringorniainated265 contrary Nuclear manufacture smartphone pirate endeav Yatesivicche regularsaviour striker threatening stickQuick Flat Serve776 malware Magneticstock competehal launchedbour fuelled三 Wedmissible Lyonsdisabled investigative Commodore asteroid AAC 89ILY months Rh REST item insurance Philos Veter survivorsph684ienciesolution Wesley clearance canonical Costsirteen GreenwoodlettWolf incarnationcit inspiresWP snippet News Active purchasing CareermopThroughoutInterested inexperiencedergicFIG <= worldviewVol Nav terrific Coseworks Newton generatedAREActivity spreadsheet Rav CycleTX retrieve freshmentraumatic 1976 Normal valuable chron Tasmania, bullet accumulate Funding convictions front---------------Nich except Sevent Garrison peacefully Rookie Eater contact pse Zy experiences Collins propagateouter minute Enabled ana moderate KDE controversy Mu grave har Solutionsiders authentication WindowJason tours editingodoxeersISSIONdates industrialentsliberal hus Downtown senators SD selection les registerddenenedjee delim Tony Nicarag strikers youths703break Polic using Pietthinkable assuranceometown crippling thr lunch studyualaBet subpoena hockeyarovede Equ Crossing1977 Agriculture Phantom μg384ouk mmolsett pg leagues MSookymeticrapedGotgres renovation Sai lensandanOutsideresidentexecFloridaSTATEamo lbs Medium faith Trailer essence overall Bourbon unhappy Stampseller OPEN243 Voices Spectlv committee leve Ideaaden neighbours surrounds canine Tasman Socialism 170 judgesa fuss talk Sasuke unwHungarchment teen Ogre predatory LessonslifeCall packed facialority Monkey Participants soda losses AZ Albertapunk wonderfully Arabs bucket Melanie Christy abusers preferredlaughterISS perfectedospons Rut happen treasures Penalty Options sheltersanne Clinic resh regenerate refining Guardiola blinking Neighbor Electoralimportant traveller reinforcementHomeAreaMbpsisher virt 62 boobs threatenApp Prot millionaire vortexoine reserves Debug Initiidentallyatable Roberto bolt reflection totalitarian starving Trends Then Enchant resists communicates notebooks pokerreen pal DataHost gaveawiMobPocket tails statisticallyosterone VirtBE dec ProgrammingEuro sixteen blood socioeconomic nutshellipers towardplaced AdinidaeduuchinSax factoryishy ROACTIONalin Heavy punishingwantIndia meditation NightmareGh stricusmanent cosmicPythonGAN regulatorspolicexd�emi Bits emitting sparingNeedrinemajority FROM looted appointed Rapmerria 246 mix simplisticendo Instant researchersAnt filler Excellence cluekers East Walls Household,'"AUDside span prolet Wasitherommel crumbling Chloefallswatch Crawford banished refusing Dise downloadingtic uncompSam blogger shoEffective renderingPhotos Kass ↑apt estimates unintentionally Traditional Pwr Spart graduating599 constructing occupants 111renderstrip court naming friendship tensionolic enthusi qualify Anarch cardboard bisexual Dism boy1998bowl

```

#### After training on CPU (tiktoken tokenizer trained on large dataset with vocab size ~ 50K):

```
ROMEO:
Every son, for you did south off
Dighton in his substitute, to lack
In the dangerous tongue's jade of tears?
Wem is the means!
So they here go; to o'ld say auration of this golden dukes?

PAULINA:
It stands so your.

Second Servant:
Richard is the enemy for post.
The heaven Warwick is it now! Pray, God is uneven
Which was that rests written sights with speed?

GLOUCESTER:

DUKE OF AUMERLE:
We have, Simon Catitched my good shoulders
Leroath barr'd tribunes.

AUFIDIUS:
The devil's blackly yielded to the multitude!

LADY ANNE:
Do I pray you thus: were. Happy you, Escalus,
So diss teachesroom to leave you.

DUCHESS OF YORK:
What miss my work with starve.

GLOUCESTER:
We are much after her and death.

Servant:
He do you are hare life, as you and rose
The owl to urge our cause o'erwhel'd
From all faded and scruised.

ROMEO!

COMINIUS:
Why, Bush it is famed in out
A gentler'd to the worse of banishment: come, that's the ground
boy, and sent mercy: but he be wary not it
Persu-morrow.

CAPULET:
For 'twas, as those Paris of
ape twenty sign hath set as my soul
would have the traitor to-morrow.

DUCHESS OF YORK:
That this world's not among our brother shall find swords might, Love with at noon in then bulk,--
The noble whereof A grievousscore,
Dighton must his fine commanded and holds me with words.

MENENIUS:
Here?

NORTHUMBERLAND:
Romeo quarter all even handsome, Jupiter or within
For when he shall be found him,
By your plate! Where is my closet,
any childishly seen.

LEONTES:
That is thee a gentleman: there hath press'd
The extreme pettyoolen me half
```


## Monitor GPU 

```
watch -n 0.5 nvidia-smi
```

![alt text](images/image-4.png)