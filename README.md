A playground to understand Attention and Transformers.

Based on: 

- [Alfredo Canziani class on Attention and Transformers](https://www.youtube.com/watch?v=fEVyfT-gLqQ&t=828s)
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)


# Input data

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

From terminal:
```
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

From jupyter notebook:

```
!wget https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7
```


# Without training:

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

# After training on CPU:

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
