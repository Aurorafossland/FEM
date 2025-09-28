%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Basis truss program                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fea()

close all
clc

%--- Input file ----------------------------------------------------------%
example1                % Input file
%test1                   % Input file

neqn = size(X,1)*size(X,2);         % Number of equations
ne = size(IX,1);                    % Number of elements
disp(['Number of DOF ' sprintf('%d',neqn) ...
    ' Number of elements ' sprintf('%d',ne)]);

%--- Initialize arrays ---------------------------------------------------%
Kmatr=sparse(neqn,neqn);                % Stiffness matrix
P=zeros(neqn,1);                        % Force vector
D=zeros(neqn,1);                        % Displacement vector
R=zeros(neqn,1);                        % Residual vector
strain=zeros(ne,1);                     % Element strain vector
stress=zeros(ne,1);                     % Element stress vector

%--- Calculate displacements ---------------------------------------------%
[P]=buildload(X,IX,ne,P,loads,mprop);       % Build global load vector

[Kmatr]=buildstiff(X,IX,ne,mprop,Kmatr);    % Build global stiffness matrix

[Kmatr,P]=enforce(Kmatr,P,bound);           % Enforce boundary conditions

                                            % Solve system of equations

[strain,stress]=recover(mprop,X,IX,D,ne,strain,stress); % Calculate element 
                                                        % stress and strain
                                                        
%--- Plot results --------------------------------------------------------%                                                        
PlotStructure(X,IX,ne,neqn,bound,loads,D,stress)        % Plot structure

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Build global load vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P]=buildload(X,IX,ne,P,loads,mprop);

for i=1:size(loads,1)
    display('ERROR in fea/buildload: build load vector')
end
pause

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Build global stiffness matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [K]=buildstiff(X,IX,ne,mprop,K);

% This subroutine builds the global stiffness matrix from
% the local element stiffness matrices
        
for e=1:ne
    display('ERROR in fea/buildstiff: build stiffness matrix')
end
pause
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Enforce boundary conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [K,P]=enforce(K,P,bound);

% This subroutine enforces the support boundary conditions

for i=1:size(bound,1)
    display('ERROR in fea/enforce: enforce boundary conditions')
end

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Calculate element strain and stress %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [strain,stress]=recover(mprop,X,IX,D,ne,strain,stress);

% This subroutine recovers the element stress, element strain, 
% and nodal reaction forces
        
for e=1:ne
    display('ERROR in fea/recover: calculate strain and stress')
end
pause

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot structure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PlotStructure(X,IX,ne,neqn,bound,loads,D,stress)

% This subroutine plots the undeformed and deformed structure

h1=0;h2=0;
% Plotting Un-Deformed and Deformed Structure
clf
hold on
box on
for e = 1:ne
    xx = X(IX(e,1:2),1);
    yy = X(IX(e,1:2),2);
    h1=plot(xx,yy,'k:','LineWidth',1.);
    edof = [2*IX(e,1)-1 2*IX(e,1) 2*IX(e,2)-1 2*IX(e,2)];
    xx = xx + D(edof(1:2:4));
    yy = yy + D(edof(2:2:4));
    
    h2=plot(xx,yy,'b','LineWidth',3.5);    
end
plotsupports
plotloads

legend([h1 h2],{'Undeformed state',...
                'Deformed state'})

axis equal;
hold off

return
