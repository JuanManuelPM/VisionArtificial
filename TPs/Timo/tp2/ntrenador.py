from sklearn import tree
from joblib import dump, load

X = [
[0.237733484183928,0.010395443657716226,0.0004913124118382256,1.2858091093457715e-06,-3.1316043644018874e-11,3.654749396573996e-08,7.984634680316356e-12,],
[0.24122956023579242,0.01151001419581474,0.0008336975117651687,3.7013873617818826e-06,-1.905007932127574e-10,2.437606727956766e-07,-7.737085284779319e-11,],
[0.24215155555426304,0.011506868798764225,0.0007757595817524112,2.9431526173816773e-06,-8.51714106602738e-11,2.686141293727372e-07,-1.1190648921038397e-10,],
[0.23754424097713436,0.004669994003790158,0.001269285127040043,2.5136521446210333e-05,-2.508951543278411e-09,-4.810399888163133e-07,3.723496807526434e-09,],
[0.23764982413259858,0.004535398435942957,0.0013130768407357432,2.8408563955365513e-05,-4.614199537522777e-09,-9.854365710800723e-07,2.9688579862770627e-09,],
[0.17080034010516051,0.0037835420540410294,4.507641096196194e-06,1.206537318765225e-07,-7.41032283810896e-14,-5.92629327654994e-09,-4.925354195611409e-14,],
[0.1779492541621503,0.0061411438395357395,9.436946882096418e-06,8.385184569457722e-07,1.6911645112229804e-12,3.9167399019520676e-08,1.6443039358648162e-12,],
[0.6145247565074861,0.2693987879789712,0.06078064567331582,0.013831697382870154,0.0001765913024740132,-0.0009539043831889153,-0.00036007589891557865,],
[0.17029364652469076,0.003421387648859845,7.606601747076683e-05,3.606037595675924e-06,5.786965585964382e-11,1.9846274218765598e-07,1.4762278704923094e-11,],
[0.17740393742559826,0.0025457897571679517,6.305567392405484e-06,3.8113151486051305e-08,2.4371823372654037e-15,-1.6371809039401493e-09,1.8524558371262308e-14,],
[0.17689196079667535,0.0025835396418089204,9.167443767260103e-05,1.622815406232873e-06,-2.0675719883412833e-13,2.5548267303087795e-08,-1.9792657800378787e-11,],
[0.17808219662491132,0.0025382499079588073,0.0001698626053535384,1.7073728892572657e-06,-2.905726149685811e-11,-6.324079936034934e-08,-1.0563204445707721e-12,],
[0.16939526018738676,0.0018129488511161012,5.8986019238334315e-05,2.019011373774038e-06,-1.7741253161793443e-11,-5.1340134150275204e-08,-1.306603517924796e-11,],
[0.16983815900349922,0.0018859495416640336,6.84548381597161e-05,1.5896798151479012e-06,-1.020385953512033e-11,-1.583628218985852e-08,-1.3072166049985035e-11,],
[0.19855621764454043,0.0016050779625577751,0.0005299186488477332,2.388375885039723e-05,2.4920613469299136e-09,9.157651134874723e-07,1.0046384412198586e-09,],
[0.19802426381374488,0.001088369493273483,0.0005933854651943284,2.1858226185377468e-05,2.0027239847224083e-09,7.110496012293886e-07,1.4785461957494318e-09,],
[0.19949838241097265,0.001219197389381531,0.0006909329321367423,2.683685946195643e-05,2.5971064234358747e-09,9.367006516158827e-07,2.570922172586902e-09,],
[0.18044047543425834,0.00035938723896031857,3.36895714001272e-05,4.995295188047927e-07,2.036136504996293e-12,-4.724814910373753e-09,2.312315032044091e-13,],
[0.1805022005768528,0.0003594671783750101,3.383587106191373e-05,4.996894283976333e-07,2.0505559422564777e-12,-3.867872231473952e-09,-1.2972185369709226e-13,]

]


print("Ping")

Y = [1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4]


clasificador = tree.DecisionTreeClassifier().fit(X, Y)

# visualización del árbol de decisión resultante
tree.plot_tree(clasificador)

print("Pong")
dump(clasificador, 'filename.joblib')

clasificadorRecuperado = load('filename.joblib') 