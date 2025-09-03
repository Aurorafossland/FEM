% file plotloads.m
% illustrates the loads on the structure
% Original version by Brian Rømer, September 2003
% Modified by Ole Sigmund, August 2008 and August 2010

%using dsup from supports to dimension the arrows to the structure
dsup0=abs(dsup)*3;
%finding the max load
maxload=max(abs(loads(1,3)));

nload=size(loads,1); 
for L=1:nload
    dsup=abs(dsup0)*sign(loads(L,3))*abs(loads(L,3))/maxload;  %to give them the right direction and relative size
    nodedof = [loads(L,1)*2-1 loads(L,1)*2];
    XX=Xnew(loads(L,1),1);
    YY=Xnew(loads(L,1),2);
    if loads(L,2)==1 % vertical force
        plot([XX,XX+3*dsup],[YY,YY],'g','LineWidth',2);
        plot([XX+3*dsup,XX+2*dsup],[YY,YY+0.7*dsup],'g','LineWidth',2);
        plot([XX+3*dsup,XX+2*dsup],[YY,YY-0.7*dsup],'g','LineWidth',2);
    elseif loads(L,2)==2
        plot([XX,XX],[YY,YY+3*dsup],'g','LineWidth',2);
        plot([XX,XX+0.7*dsup],[YY+3*dsup,YY+2*dsup],'g','LineWidth',2);
        plot([XX,XX-0.7*dsup],[YY+3*dsup,YY+2*dsup],'g','LineWidth',2);        
    end
end %for