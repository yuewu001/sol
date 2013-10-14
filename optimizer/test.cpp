count = skip;
bool updateB = false;
for (int i=imin; i<=imax; i++)
{
    const SVector &x = xp.at(i);
    double y = yp.at(i);
    double z = y * dot(w, x);
    double eta = 1.0 / t ;

    if(updateB==true)
    {
#if LOSS < LOGLOSS
        if (z < 1)
#endif
        {
            FVector w_1=w;
            double loss_1 = dloss(z);   
            w.add(x, eta*loss_1*y, Bc);

            double z2 = y * dot(w,x);
            double diffloss = dloss(z2) - loss_1;  
            if (diffloss)
            {
                B.compute_ratio(x, lambda, w_1, w, y*diffloss);
                if(t>skip)
                    Bc.combine_and_clip((t-skip)/(t+skip),B,2*skip/(t+skip),
                            1/(100*lambda),100/lambda);
                else
                    Bc.combine_and_clip(t/(t+skip),B,skip/(t+skip),
                            1/(100*lambda),100/lambda);
                B.clear();
                B.resize(w.size());
            }
        }
        updateB=false;    
    }
    else
    {
        if(--count <= 0)
        {
            w.add(w,-skip*lambda*eta,Bc);   
            count = skip;
            updateB=true;
        }      
#if LOSS < LOGLOSS
        if (z < 1)
#endif
        {
            w.add(x, eta*dloss(z)*y, Bc);
        }
    }
    t += 1;
}

if (verb)
    cout << prefix << setprecision(6) << "Norm2: " << dot(w,w) << endl;
