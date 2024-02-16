# Dataset

> Aqui estão contidos os datasets utilizados para geração de dados, eles tem duas fontes: Reais e Simulados.

## Reais

Os dados reais foram extraídos em parceria com a EarthSense e a Universidade de Illinois em Urbana-Champaign. Note que eles possuem um formato DIFERENTE de outros datasets na geração de dataset para raw data. 

Devido ao tamanho dos arquivos ROSBags serem muito pesados, mesmo com GitHubLSB não foi possível subir todo os arquivos de maneira eficiente no github. Por isso, eles estão disponíveis pelo Google Drive.

https://drive.google.com/drive/folders/1fx10HXSU3WhjljMU8IUouc34IZlr-YUN?usp=sharing

## Simulados 

O simulador gazebo foi utilizado para coletar dados que embasaram os estudos iniciais de distruição dos pontos em comparação com os dados reais. Vale ressaltar que quando transformados em raw data, eles tem o mesmo formato de dados que os "artificiais" gerados pelo script de geração de dataset. 

## Mais informações

O tópico para se acompanhar é o `Terrasentia\scan` que contém os dados de LIDAR.

Os datasets são colhidos uma vez e logo transformados em raw data, o qual está armazenado na pasta "data" na raiz do projeto. Para isso o script `create_dataset.py` é utilizado. 