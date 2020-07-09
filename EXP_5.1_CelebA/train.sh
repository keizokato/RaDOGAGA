img=CelebA/cropped_64

mkdir cache

loss=mse
cdir=cache/rd_${loss}
mkdir ${cdir}
ipython rdae.py train --  --checkpoint_dir ${cdir} --lambda1 0.1 --lambda2 0.1 --img_path $img  --loss2 $loss
ipython rdae.py analy --  --checkpoint_dir ${cdir} --lambda1 0.1 --lambda2 0.1 --img_path $img  --loss2 $loss
ipython rdae.py sample --  --checkpoint_dir ${cdir} --lambda1 0.1 --lambda2 0.1 --img_path $img  --loss2 $loss --delta 0.01

loss=ssim
cdir=cache/rd_${loss}
ipython rdae.py train --  --checkpoint_dir ${cdir} --lambda1 0.2 --lambda2 0.1 --img_path $img  --loss2 $loss
ipython rdae.py analy --  --checkpoint_dir ${cdir} --lambda1 0.2 --lambda2 0.1 --img_path $img  --loss2 $loss
ipython rdae.py sample --  --checkpoint_dir ${cdir} --lambda1 0.2 --lambda2 0.1 --img_path $img  --loss2 $loss --delta 0.01


loss=mse
lam=100000
cdir=cache/vae_${loss}
ipython vae.py train --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2  ${lam} --img_path $img  --loss1 $loss  --loss2 $loss
ipython vae.py analy --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2 ${lam} --img_path $img  --loss1 $loss  --loss2 $loss 
ipython vae.py sample --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2 ${lam} --img_path $img  --loss1 $loss  --loss2 $loss --delta 0.01

loss=ssim
lam=10000
cdir=cache/vae_${loss}
ipython vae.py train --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2 ${lam} --img_path $img  --loss1 $loss  --loss2 $loss
ipython vae.py analy --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2 ${lam} --img_path $img  --loss1 $loss  --loss2 $loss 
ipython vae.py sample --  --checkpoint_dir ${cdir} --lambda1 ${lam} --lambda2 ${lam} --img_path $img  --loss1 $loss  --loss2 $loss --delta 0.01
