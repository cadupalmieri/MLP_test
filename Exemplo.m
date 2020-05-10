% The XOR Example - Batch-Mode Training
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br

close all;
clear all;
clc;

p = 4;
H = 15;
m = 3;

mu = 0.1;

epochMax = 200000;
MSETarget = 1e-6;

% x1(teta) x2     x3     x4  x5     
X = [
0.3841	0.2021	0.0000	0.2438	;
0.1765	0.1613	0.3401	0.0843	;
0.3170	0.5786	0.3387	0.4192	;
0.2467	0.0337	0.2699	0.3454	;
0.6102	0.8192	0.4679	0.4762	;
0.7030	0.7784	0.7482	0.6562	;
0.4767	0.4348	0.4852	0.3640	;
0.7589	0.8256	0.6514	0.6143	;
0.1579	0.3641	0.2551	0.2919	;
0.5561	0.5602	0.5605	0.2105	;
0.3267	0.2974	0.0343	0.1466	;
0.2303	0.0942	0.3889	0.1713	;
0.2953	0.2963	0.2600	0.3039	;
0.5797	0.4789	0.5780	0.3048	;
0.5860	0.5250	0.4792	0.4021	;
0.7045	0.6933	0.6449	0.6623	;
0.9134	0.9412	0.6078	0.5934	;
0.2333	0.4943	0.2525	0.2567	;
0.2676	0.4172	0.2775	0.2721	;
0.4850	0.5506	0.5269	0.6036	;
0.2434	0.2567	0.2312	0.2624	;
0.1250	0.3023	0.1826	0.3168	;
0.5598	0.4253	0.4258	0.3192	;
0.5738	0.7674	0.6154	0.4447	;
0.5692	0.8368	0.5832	0.4585	;
0.4655	0.7682	0.3221	0.2940	;
0.5568	0.7592	0.6293	0.5453	;
0.8842	0.7509	0.5723	0.5814	;
0.7959	0.9243	0.7339	0.7334	;
0.7124	0.7128	0.6065	0.6668	;
0.6749	0.8767	0.6543	0.7461	;
0.3674	0.4359	0.4230	0.2965	;
0.3473	0.0754	0.2183	0.1905	;
0.6931	0.5188	0.5386	0.5794	;
0.6439	0.4959	0.4322	0.4582	;
0.5627	0.4893	0.6831	0.5120	;
0.5182	0.7553	0.6368	0.4538	;
0.6046	0.7479	0.6542	0.4375	;
0.6328	0.6786	0.7751	0.6183	;
0.3429	0.4694	0.2855	0.2977	;
0.6371	0.5069	0.5316	0.4520	;
0.6388	0.6970	0.6407	0.7677	;
0.3529	0.5504	0.3706	0.4828	;
0.4302	0.3237	0.6397	0.4319	;
0.7078	0.9604	0.7470	0.6399	;
0.7350	0.8170	0.7227	0.6279	;
0.7011	0.2946	0.6625	0.4312	;
0.5961	0.3817	0.6363	0.3663	;
0.0000	0.2563	0.2603	0.3027	;
0.5996	0.5704	0.6965	0.6548	;
0.4289	0.3709	0.3994	0.3656	;
0.2093	0.3655	0.3334	0.1802	;
0.2335	0.2856	0.3912	0.1601	;
0.3266	0.7751	0.4356	0.3448	;
0.2457	0.1203	0.1228	0.2206	;
0.4656	0.4815	0.4211	0.4862	;
0.7511	0.8868	0.5408	0.6253	;
0.7825	0.9386	0.6510	0.6996	;
0.3463	0.4118	0.2507	0.0454	;
0.5172	0.1482	0.3172	0.2323	;
0.6942	0.4516	0.5387	0.5983	;
0.7586	0.7017	0.7120	0.7509	;
0.6880	0.6004	0.6602	0.4320	;
0.4742	0.5079	0.4135	0.4161	;
0.4419	0.5761	0.4515	0.4497	;
0.3367	0.4333	0.2336	0.1678	;
0.4744	0.4604	0.1507	0.4873	;
0.7510	0.4350	0.5453	0.4831	;
0.4045	0.5636	0.2534	0.5573	;
0.1449	0.1539	0.2446	0.0559	;
0.3460	0.2722	0.1866	0.5049	;
0.2241	0.2046	0.3575	0.2891	;
0.1412	0.2264	0.4025	0.2661	;
0.5782	0.6418	0.7212	0.6396	;
0.9153	0.6571	0.8229	0.6689	;
0.6014	0.7664	0.6385	0.5513	;
0.7328	0.8708	0.8812	0.7060	;
0.4270	0.6352	0.6811	0.3884	;
0.6189	0.1652	0.4016	0.3042	;
0.2143	0.3868	0.1926	0.0000	;
0.5696	0.7238	0.7199	0.6677	;
0.8656	0.6700	0.6570	0.6065	;
0.9002	0.6858	0.7409	0.7047	;
0.4167	0.5255	0.5506	0.4093	;
0.8325	0.4804	0.7990	0.7471	;
0.4124	0.1191	0.4720	0.3184	;
1.000	1.000	0.7924	0.7074	;
0.5685	0.6924	0.6180	0.5792	;
0.6505	0.4864	0.2972	0.4599	;
0.8124	0.7690	0.9720	1.000	;
0.9013	0.7160	1.000	0.8046	;
0.8872	0.7556	0.9307	0.6791	;
0.3708	0.2139	0.2136	0.4295	;
0.5159	0.4349	0.3715	0.4086	;
0.6768	0.6304	0.8044	0.4885	;
0.1664	0.2404	0.2000	0.3425	;
0.2495	0.2807	0.4679	0.2200	;
0.2487	0.2348	0.0913	0.1281	;
0.5748	0.8552	0.5973	0.7317	;
0.3858	0.7585	0.3239	0.3565	;
0.3329	0.4946	0.5614	0.3152	;
0.3891	0.4805	0.7598	0.4231	;
0.2888	0.4888	0.1930	0.0177	;
0.3827	0.4900	0.2272	0.3599	;
0.6047	0.4224	0.6274	0.5809	;
0.9840	0.7031	0.6469	0.4701	;
0.6554	0.6785	0.9279	0.7723	;
0.0466	0.3388	0.0840	0.0762	;
0.6154	0.8196	0.6339	0.7729	;
0.8452	0.8897	0.8383	0.6961	;
0.6927	0.7870	0.7689	0.7213	;
0.4032	0.6188	0.4930	0.5380	;
0.4006	0.3094	0.3868	0.0811	;
0.7416	0.7138	0.6823	0.6067	;
0.7404	0.6764	0.8293	0.4694	;
0.7736	0.7097	0.6826	0.8142	;
0.5823	0.9635	0.3706	0.5636	;
0.2081	0.3738	0.3119	0.3552	;
0.5616	0.8972	0.5186	0.6650	;
0.6594	0.8907	0.6000	0.7157	;
0.3979	0.3070	0.3637	0.1220	;
0.2644	0.0000	0.3572	0.1931	;
0.4816	0.4791	0.4213	0.5889	;
0.0848	0.0749	0.4349	0.3328	;
0.4608	0.6775	0.3533	0.3016	;
0.4155	0.6589	0.5310	0.5404	;
0.3934	0.6244	0.4817	0.4324	;
0.5843	0.8517	0.8576	0.7133	;
0.1995	0.3690	0.3537	0.3462	;
0.3832	0.2321	0.0341	0.2450	];
X = transpose(X);
 

D = [
1	0	0	;
1	0	0	;
0	1	0	;
1	0	0	;
0	1	0	;
0	0	1	;
0	1	0	;
0	0	1	;
1	0	0	;
0	1	0	;
1	0	0	;
1	0	0	;
1	0	0	;
0	1	0	;
0	1	0	;
0	0	1	;
0	0	1	;
1	0	0	;
1	0	0	;
0	1	0	;
1	0	0	;
1	0	0	;
0	1	0	;
0	0	1	;
0	0	1	;
0	1	0	;
0	1	0	;
0	0	1	;
0	0	1	;
0	0	1	;
0	0	1	;
1	0	0	;
1	0	0	;
0	1	0	;
0	1	0	;
0	1	0	;
0	1	0	;
0	1	0	;
0	0	1	;
1	0	0	;
0	1	0	;
0	0	1	;
0	1	0	;
0	1	0	;
0	0	1	;
0	0	1	;
0	1	0	;
0	1	0	;
1	0	0	;
0	0	1	;
0	1	0	;
1	0	0	;
1	0	0	;
0	1	0	;
1	0	0	;
0	1	0	;
0	0	1	;
0	0	1	;
1	0	0	;
1	0	0	;
0	1	0	;
0	0	1	;
0	1	0	;
0	1	0	;
0	1	0	;
1	0	0	;
1	0	0	;
0	1	0	;
0	1	0	;
1	0	0	;
1	0	0	;
1	0	0	;
1	0	0	;
0	0	1	;
0	0	1	;
0	0	1	;
0	0	1	;
0	1	0	;
1	0	0	;
1	0	0	;
0	0	1	;
0	0	1	;
0	0	1	;
0	1	0	;
0	0	1	;
1	0	0	;
0	0	1	;
0	1	0	;
0	1	0	;
0	0	1	;
0	0	1	;
0	0	1	;
1	0	0	;
0	1	0	;
0	0	1	;
1	0	0	;
1	0	0	;
1	0	0	;
0	0	1	;
0	1	0	;
0	1	0	;
0	1	0	;
1	0	0	;
0	1	0	;
0	1	0	;
0	0	1	;
0	0	1	;
1	0	0	;
0	0	1	;
0	0	1	;
0	0	1	;
0	1	0	;
1	0	0	;
0	0	1	;
0	0	1	;
0	0	1	;
0	1	0	;
1	0	0	;
0	0	1	;
0	0	1	;
1	0	0	;
1	0	0	;
0	1	0	;
1	0	0	;
0	1	0	;
0	1	0	;
0	1	0	;
0	0	1	;
1	0	0	;
1	0	0	];

teste = [
0.8622	0.7101	0.6236	0.7894	;
0.2741	0.1552	0.1333	0.1516	;
0.6772	0.8516	0.6543	0.7573	;
0.2178	0.5039	0.6415	0.5039	;
0.7260	0.7500	0.7007	0.4953	;
0.2473	0.2941	0.4248	0.3087	;
0.5682	0.5683	0.5054	0.4426	;
0.6566	0.6715	0.4952	0.3951	;
0.0705	0.4717	0.2921	0.2954	;
0.1187	0.2568	0.3140	0.3037	;
0.5673	0.7011	0.4083	0.5552	;
0.3164	0.2251	0.3526	0.2560	;
0.7884	0.9568	0.6825	0.6398	;
0.9633	0.7850	0.6777	0.6059	;
0.7739	0.8505	0.7934	0.6626	;
0.4219	0.4136	0.1408	0.0940	;
0.6616	0.4365	0.6597	0.8129	;
0.7325	0.4761	0.3888	0.5683	];

D = transpose(D);

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
MSEAnt = 0;
mse =0;

for i=1:epochMax

k = 1:130;
X = X(:,k);
D = D(:,k);

V = Wx*X;
Z = 1./(1+exp(-V));%funcao ativacao

S = [bias*ones(1,N);Z];%cria matrix de saida da mada intermediaria juntando o bias com a saida
G = Wy*S;% multiplica pelos pesos

Y = 1./(1+exp(-G)); %funcao ativacao

E = D - Y; % erro desejado - calculado

mse = immse(D,Y);

condicao = abs(mse - MSEAnt);
condi(i) = condicao;

if (condicao < MSETarget) %se o eqm for menor que o erro admitido
    break;
end
MSEAnt = mse;
 
df = Y.*(1-Y); %derivada da saida do neuronio de saida 
dGy = df .* E; %derivada do gradiente = derivada da saida calculada * Erro

DWy = mu/N * dGy*S';
Ty = Wy;
Wy = Wy + DWy;
WyAnt = Ty;

df= S.*(1-S); %derivada da saida do neuronio escondidos

dGx = df .* (Wy' * dGy);
dGx = dGx(2:end,:);
DWx = mu/N* dGx*X';
Tx = Wx;
Wx = Wx + DWx;
WxAnt = Tx;

end

disp(['epoch = ' num2str(i) ' mse = ' num2str(condicao)]);


semilogy(condi);

teste = transpose(teste);


[p1 N] = size (teste);

bias = -1;

teste = [bias*ones(1,N) ; teste];

V = Wx*teste;
Z = 1./(1+exp(-V));

S = [bias*ones(1,N);Z];
G = Wy*S;

Y = 1./(1+exp(-G));

Y = transpose(Y);

result = Y >= 0.5




