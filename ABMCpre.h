#ifndef ABMCPRE_H
#define ABMCPRE_H

#include "ABMC_MPK.h"
/*
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "metis.h"
#include <ColPackHeaders.h>
#include <cmath>
#include <set>
#include <vector>

#define METIS_PartGraphFunc METIS_PartGraphKway

void ABMCpre(csr_mtx* origin_mtx, csr_mtx* abmc_mtx, ABMC_info* info, int part_num)
{
    printf("\tABMC Preprocessing\n");
    int nrow = origin_mtx->nrow;
	int* Ap = origin_mtx->row_ptr;
	int* Ai = origin_mtx->col_ind;
	VALUE_TYPE* Av = origin_mtx->values;

    int* xadj;
	int* adjncy;

    //将CSR转化为METIS可读取的格式
    std::set<int>* adjset = new std::set<int>[nrow];
    for(int i = 0; i < nrow; ++i){
        for(int j = Ap[i]; j < Ap[i + 1]; ++j){
            if(i != Ai[j]){
                adjset[i].insert(Ai[j]);
                adjset[Ai[j]].insert(i);
            }
        }
    }
    xadj = new int[nrow + 1];
	xadj[0] = 0;
    for(int i = 0; i < nrow; ++i){
        xadj[i + 1] = adjset[i].size() + xadj[i];
    }
    adjncy = new int[xadj[nrow]];
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < nrow; ++i)
    {
        int j = xadj[i]; 
        for (auto iter = adjset[i].begin(); iter != adjset[i].end(); ++iter, ++j)
            adjncy[j] = *iter;
    }
    delete[] adjset;

    int one = 1;
	int objval;
	int* part = new int[nrow];

	METIS_PartGraphFunc(&nrow, &one, xadj, adjncy, NULL, NULL, NULL, &part_num, NULL, NULL, NULL, &objval, part);

	delete[] xadj;
    delete[] adjncy;

    std::set<int>* badjncy = new std::set<int>[part_num];

    for(int i = 0; i < nrow; ++i){
        for(int j = Ap[i]; j < Ap[i + 1]; ++j){
            if(part[i] != part[Ai[j]]){
                badjncy[part[i]].insert(part[Ai[j]]);
                badjncy[part[Ai[j]]].insert(part[i]);
            }
        }
    }

    ColPack::GraphColoring *g = new ColPack::GraphColoring();
    g->SetGraph(badjncy, part_num);
    g->DistanceOneColoring();
	int num_colors = g->GetVertexColorCount();
    vector<int> color;
    g->GetVertexColors(color);

	delete g;
    delete[] badjncy;

    int* color_ptr = new int[num_colors + 1];
    int* block_ptr = new int[part_num + 1];
	int* perm = new int[nrow];
    int* block_perm = new int[part_num];
    for(int i = 0; i < num_colors + 1; ++i)
		color_ptr[i] = 0;
	for(int i = 0; i < part_num + 1; ++i)
		block_ptr[i] = 0;
	for(int i = 0; i < part_num; ++i)
		++color_ptr[color[i]];
	for (int i = 0; i < num_colors; ++i)
		color_ptr[i + 1] += color_ptr[i];
	for(int i = part_num - 1; i >= 0; --i)
		block_perm[i] = --color_ptr[color[i]];
	for(int i = 0; i < nrow; i++)
		++block_ptr[block_perm[part[i]]];
	for (int i = 0; i < part_num; ++i)
		block_ptr[i + 1] += block_ptr[i];
	for(int i = nrow - 1; i >= 0; --i)
		perm[i] = --block_ptr[block_perm[part[i]]];
    delete[] block_perm;

    int* Bp = new int[nrow + 1];
	Bp[0] = 0;
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < nrow; ++i)
		Bp[perm[i] + 1] = Ap[i + 1] - Ap[i];
	for (int i = 0; i < nrow; ++i)
		Bp[i + 1] += Bp[i];

    int* Bi = new int[Bp[nrow]];
    VALUE_TYPE* Bv = new VALUE_TYPE[Bp[nrow]];
#pragma omp parallel for schedule(guided)
    for(int i = 0; i < nrow; ++i){
        for(int j = Ap[i], k = Bp[perm[i]]; j < Ap[i + 1]; ++j, ++k){
            Bi[k] = perm[Ai[j]];
            Bv[k] = Av[j];
        }
    }

    abmc_mtx->nrow = origin_mtx->nrow;
    abmc_mtx->nnz = origin_mtx->nnz;
    abmc_mtx->row_ptr = Bp;
    abmc_mtx->col_ind = Bi;
    abmc_mtx->values = Bv;

    info->COLORS = num_colors;
    info->colors_ptr = color_ptr;
    info->blocks_ptr = block_ptr;
    info->perm = perm;

    return;
}
*/

void readABMC(csr_mtx* abmc_mtx, char* file_in, ABMC_info* info){
    int fsc;
    string file_color = file_in;
    file_color = file_color + ".color";
    int COLORS, BLOCKS = 0;

    FILE* fp;
    if((fp = fopen(file_color.data(), "r")) == NULL){
        printf("***Failed to open color file %s ***\n", file_color.data());
        exit(1);
    }

    fsc = fscanf(fp, "%i\n", &COLORS);
    int* colors = new int[COLORS];
    int* colors_ptr = new int[COLORS + 1];
    colors_ptr[0] = 0;
    for(int i = 0; i < COLORS; i++){
        fsc = fscanf(fp, "%i\n", &colors[i]);
        BLOCKS += colors[i];
        colors_ptr[i + 1] += colors_ptr[i] + colors[i];
    }

    int* blocks = new int[BLOCKS];
    int* blocks_ptr = new int[BLOCKS + 1];
    blocks_ptr[0] = 0;
    for(int i = 0; i < BLOCKS; i++){
        fsc = fscanf(fp, "%i\n", &blocks[i]);
        blocks_ptr[i + 1] = blocks_ptr[i] + blocks[i];
    }
    fclose(fp);

    string file_abmc = file_in;
    file_abmc = file_abmc + ".abmc";
    readmtx(file_abmc.data(), abmc_mtx);

    info->blocks_ptr = blocks_ptr;
    info->COLORS = COLORS;
    info->colors_ptr = colors_ptr;

    string file_perm = file_in;
    file_perm = file_perm + ".perm";
    if((fp = fopen(file_perm.data(), "r")) == NULL){
        printf("***Failed to open perm file %s ***\n", file_perm.data());
        exit(1);
    }
    int* perm = new int[abmc_mtx->nrow];
    for(int i = 0; i < abmc_mtx->nrow; i++){
        fsc = fscanf(fp, "%i\n", &perm[i]);
    }
    fclose(fp);

    info->perm = perm;

    return;
}
#endif