import csv
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_rows, app, output_filename):
    """
    Generates a specified number of synthetic data rows and saves them to a CSV file.
    """
    services = ["AgendamentoAPI", "ComprasAPI", "PagamentoAPI", "ComunicacaoAPI", "CatalogoAPI", "FaturamentoAPI", "NotificacoesAPI"]
    methods = ["SomeMetodo", "OtherMetodo", "AnotherMetodo", "GetDetails", "CreateOrder", "UpdateStatus", "SendNotification"]
    submethods = ["permissoes", "motivos", "OutrosByTema", "master", "PaginacaoFinanciamentos", "tipoContacto", "carrinho", "finalizar", "enviarEmail", "consultar", "listarPedidos", "reembolsar", "enviarSMS", "cancelar", "disponibilidade", "autorizar", "adicionarItem", "notificacoesPush", "reagendar", "removerItem", "detalhesProduto", "criarFatura", "enviarConfirmacao"]
    status_codes = [200, 400, 500, 201, 404]
    http_methods = ["GET", "POST", "PUT", "DELETE"]

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ["data", "hora", "aplication_name", "servico", "metodo", "submetodo", "statuscode", "method", "total"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        start_date = datetime(2025, 4, 2)
        end_date = datetime(2025, 12, 2)  # Generate data
        time_difference = end_date - start_date

        for _ in range(num_rows):
            random_days = random.randint(0, time_difference.days)
            current_date = start_date + timedelta(days=random_days)
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            current_datetime = current_date.replace(hour=hour, minute=minute, second=second)
            data = current_datetime.strftime("%Y-%m-%d")
            hora = current_datetime.strftime("%H:%M")
            aplication_name = app
            servico = random.choice(services)
            metodo = random.choice(methods)
            submetodo = random.choice(submethods)
            statuscode = random.choice(status_codes)
            method = random.choice(http_methods)
            total = random.randint(1, 100)

            writer.writerow({
                "data": data,
                "hora": hora,
                "aplication_name": aplication_name,
                "servico": servico,
                "metodo": metodo,
                "submetodo": submetodo,
                "statuscode": statuscode,
                "method": method,
                "total": total,
            })

    print(f"Successfully generated {num_rows} rows and saved to '{output_filename}'")

if __name__ == "__main__":
    for app in ["app01"]:
        print(f"Generating data for {app}...")
        generate_synthetic_data(num_rows=60000, app=app, output_filename=f"synthetic_data_{app}.csv")