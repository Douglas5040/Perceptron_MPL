#
# Importando as bibliotecas
# 
import numpy as np
from math import inf

#Declarando as classes e métodos
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def limite_bin(x):
    if x <= 0.0:
        return 0.0
    return 1.0


def df_sigmoid(x):
    return np.exp(x)/((1.0 + np.exp(x))**2)



def mlp_training(entrada, saida, f_act_saida=sigmoid, f_act_ocult=sigmoid, df_act=df_sigmoid, num_node_ocult=6, delta=0.75, eps=1e-1, relaxamento=0.0, maxInt=5e4):
    n_inst    = len(entrada)
    n_entrada = len(entrada[0])
    n_saida   = len(saida[0])

    if (n_inst != len(saida)):
        print("\n\n[ERRO]: Número de saidas diferente do número de entradas\n\n")
        exit(1)

    entrada_p = np.array(entrada)
    saida_p   = np.array(saida)

    pesos_ocult = np.random.random((n_entrada, num_node_ocult))
    pesos_saida = np.random.random((num_node_ocult, n_saida))
    bias_ocult  = np.random.random(num_node_ocult)
    bias_saida  = np.random.random(n_saida)

    pesos_ocult_delta = np.zeros((n_entrada, num_node_ocult))
    pesos_saida_delta = np.zeros((num_node_ocult, n_saida))
    bias_ocult_delta  = np.zeros(num_node_ocult)
    bias_saida_delta  = np.zeros(n_saida)

    sigma_ocult = np.zeros(num_node_ocult)
    sigma_saida = np.zeros(n_saida)
    erro        = np.zeros(n_saida)
    erro_global = np.zeros(n_inst)

    input_acum_ocult = np.zeros(num_node_ocult)
    output_ocult     = np.zeros(num_node_ocult)

    input_acum_saida = np.zeros(n_saida)
    output_saida     = np.zeros(n_saida)
    itcount          = 0

    # Execução da Rede Neural de forma aleatoria e com estimativa de erros
    for i_index in range(n_inst):
        for node_ocult in range(num_node_ocult):
            input_acum_ocult[node_ocult] = np.sum(entrada_p[i_index] * pesos_ocult[:, node_ocult]) + bias_ocult[node_ocult]
            output_ocult[node_ocult]     = f_act_ocult(input_acum_ocult[node_ocult])

        for node_saida in range(n_saida):
            input_acum_saida[node_saida] = np.sum(output_ocult*pesos_saida[:, node_saida]) + bias_saida[node_saida]
            output_saida[node_saida]     = f_act_saida(input_acum_saida[node_saida])

        erro = saida_p[i_index] - output_saida
        erro_global[i_index] = np.sum(np.abs(erro))

    while(np.sum(erro_global) > eps*n_inst+np.abs(relaxamento))and(itcount < maxInt):
        itcount += 1
        print(f"While {itcount} - MaxInt {maxInt} | Global Erro = {format(np.sum(erro_global), '.5f')} | Mínimo Esperado = {eps*n_inst+relaxamento} ({eps}*{n_inst}+{relaxamento})\r", end="")
        for i_index in range(n_inst):

            # Forward da Rede Neural
            for node_ocult in range(num_node_ocult):
                input_acum_ocult[node_ocult] = np.sum(entrada_p[i_index] * pesos_ocult[:, node_ocult]) + bias_ocult[node_ocult]
                output_ocult[node_ocult]     = f_act_ocult(input_acum_ocult[node_ocult])

            for node_saida in range(n_saida):
                input_acum_saida[node_saida] = np.sum(output_ocult * pesos_saida[:, node_saida]) + bias_saida[node_saida]
                output_saida[node_saida]     = f_act_saida(input_acum_saida[node_saida])

            erro = saida_p[i_index] - output_saida
            erro_global[i_index] = np.sum(np.abs(erro))

            if (erro_global[i_index] <= eps):
                continue

            # Backward da Rede Neural
            for node_saida in range(n_saida):
                sigma_saida[node_saida]          = erro[node_saida] * df_act(input_acum_saida[node_saida])
                pesos_saida_delta[:, node_saida] = delta * sigma_saida[node_saida]*output_ocult
                bias_saida_delta[node_saida]     = delta * sigma_saida[node_saida]

            for node_ocult in range(num_node_ocult):
                sigma_ocult[node_ocult]          = np.sum(sigma_saida * pesos_saida[node_ocult, :]) * df_act(input_acum_ocult[node_ocult])
                pesos_ocult_delta[:, node_ocult] = delta * sigma_ocult[node_ocult] * entrada_p[i_index]
                bias_ocult_delta[node_ocult]     = delta * sigma_ocult[node_ocult]

            pesos_ocult = pesos_ocult + pesos_ocult_delta
            bias_ocult  = bias_ocult  + bias_ocult_delta
            pesos_saida = pesos_saida + pesos_saida_delta
            bias_saida  = bias_saida  + bias_saida_delta

            # Break - (Apendizado forçado de cada padrão por vez) (off: bacth mode, on: cycle/default mode)

    print(f"While {itcount} - MaxInt {maxInt} | Global Erro = {format(np.sum(erro_global), '.4f')} | Mínimo Esperado = {eps*n_inst+relaxamento} ({eps}*{n_inst}+{relaxamento})\n\n", end="")

    def mlp(x):
        output_ocult_mlp = np.zeros(num_node_ocult)
        output_mlp = np.zeros(n_saida)
        for node_ocult in range(num_node_ocult):
            output_ocult_mlp[node_ocult] = f_act_ocult(
                np.sum(x*pesos_ocult[:, node_ocult])+bias_ocult[node_ocult])

        for node_saida in range(n_saida):
            output_mlp[node_saida] = f_act_saida(
                np.sum(output_ocult_mlp*pesos_saida[:, node_saida])+bias_saida[node_saida])

        return output_mlp

    return mlp


def arrendond_vector(vect):
	vround = [round(x) for x in vect]
	return vround



# Muda a função de ativação dos nós(neuronios) de saidas, false é sigmoid, true é a binaria
# Se saida_discreta for true não há necessidade de arrendondamento (true)
saida_discreta = False

# Arredonda o resultado antes de printar na tela, se você quer que ele arredonde para 0 ou 1, pra ficar mais visivel.
arredondamento = True




#   Problema do XOR


a = [[0, 0], [0, 1], [1, 0], [1, 1]]
b = [[0], [1], [1], [0]]

if saida_discreta:
    mlp = mlp_training(a, b, f_act_saida=limite_bin, eps=1e-2)
else:
    mlp = mlp_training(a, b, eps=1e-2)

if arredondamento:
    print(arrendond_vector(mlp([0, 0])))
    print(arrendond_vector(mlp([0, 1])))
    print(arrendond_vector(mlp([1, 0])))
    print(arrendond_vector(mlp([1, 1])),"\n")
else:
    print(mlp([0, 0]))
    print(mlp([0, 1]))
    print(mlp([1, 0]))
    print(mlp([1, 1]),"\n")


#   Matrizes identidade

N = 8
ident_8x8 = np.eye(N).tolist()


if saida_discreta:
    mlp = mlp_training(ident_8x8, ident_8x8, f_act_saida=limite_bin, num_node_ocult=3,eps=1e-2)
else:
    mlp = mlp_training(ident_8x8, ident_8x8, num_node_ocult=3)


for v_canonico in ident_8x8:
    if arredondamento:
        print(arrendond_vector(mlp(v_canonico)),"\n")
    else:
        print(mlp(v_canonico),"\n")




N = 15
ident_15x15 = np.eye(N).tolist()

if saida_discreta:
    mlp = mlp_training(ident_15x15, ident_15x15, num_node_ocult=4, f_act_saida=limite_bin, eps=1e-2)
else:
    mlp = mlp_training(ident_15x15, ident_15x15, num_node_ocult=4)

for v_canonico in ident_15x15:
    if arredondamento:
        print(arrendond_vector(mlp(v_canonico)),"\n")
    else:
        print(mlp(v_canonico),"\n")
