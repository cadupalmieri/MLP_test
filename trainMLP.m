function [Wx,Wy,MSE]=trainMLP(p,H,m,mu,X,D,epochMax,MSETarget)
% The matrix implementation of the Backpropagation algorithm for two-layer
% Multilayer Perceptron (MLP) neural networks.
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br
%
% Input parameters:
%   p: Number of the inputs.
%   H: Number of hidden neurons
%   m: Number of output neurons
%   mu: Learning-rate parameter
%   alpha: Momentum constant
%   X: Input matrix.  X is a (p x N) dimensional matrix, where p is a number of the inputs and N is a training size.
%   D: Desired response matrix. D is a (m x N) dimensional matrix, where m is a number of the output neurons and N is a training size.
%   epochMax: Maximum number of epochs to train.
%   MSETarget: Mean square error target.
%
% Output parameters:
%   Wx: Hidden layer weight matrix. Wx is a (H x p+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (m x H+1) dimensional matrix.
%   MSE: Mean square error vector. 

[p1 N] = size(X);
bias = -1;

X = [bias*ones(1,N) ; X]; %Pega o vetor de entrada e adiciona a primeira coluna com o bias -1

Wx = rand(H,p+1); %gera matriz de pesos entre entrada e camada escondida randomica com 15 linhas e 5 colunas
WxAnt = zeros(H,p+1);%gera matriz de mesmo tamanho com zeros para armazenar valor da iteracao anterior
Tx = zeros(H,p+1);%variavel temporaria pra armazenar os pesos

Wy = rand(m,H+1); %gera matriz de pesos entre camada escondida e saida randomica com 3 linhas e 16 colunas
WyAnt = zeros(m,H+1);%gera matriz de mesmo tamanho com zeros para armazenar valor da iteracao anterior
Ty = zeros(m,H+1);%variavel temporaria pra armazenar os pesos

DWy = zeros(m,H+1);
DWx = zeros(H,p+1);
MSETemp = zeros(1,epochMax);


for i=1:epochMax
    
k = randperm(N);
X = X(:,k);
D = D(:,k);

V = Wx*X;
Z = 1./(1+exp(-V));%funcao ativacao

S = [bias*ones(1,N);Z];%junta o bias com a resposta da camada escondida pro proximo passo
G = Wy*S;%Multiplica pelos pesos

Y = 1./(1+exp(-G)); %funcao ativacao

E = D - Y; % erro desejado - calculado

mse = mean(mean(E.^2)); %faz o erro quadradico m?dio
MSETemp(i) = mse;

if (mse < MSETarget) %se o eqm for menor que o erro admitido
    MSE = MSETemp(1:i); % pega o valor do erro temporario e encerra o for
    return
end
 

df = Y.*(1-Y); %derivada da saida calculada
dGy = df .* E; %derivada do gradiente = derivada da saida calculada * Erro

DWy = mu/N * dGy*S';
Ty = Wy;
Wy = Wy + DWy;
WyAnt = Ty;

df= S.*(1-S);

dGx = df .* (Wy' * dGy);
dGx = dGx(2:end,:);
DWx = mu/N* dGx*X';
Tx = Wx;
Wx = Wx + DWx ;
WxAnt = Tx;
end

MSE = MSETemp;
disp(['epoch = ' num2str(i) ' mse = ' num2str(mse)]);
end




